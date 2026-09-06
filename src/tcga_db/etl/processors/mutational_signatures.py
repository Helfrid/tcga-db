"""Mutational signatures processor — COSMIC SBS exposures per patient.

Loads per-patient COSMIC SBS mutational signature exposures from a
wide-format TSV (samples × signatures) and melts to long format for the
mutational_signatures table.

Expected source file format (TSV, one row per sample):

    Sample  SBS1  SBS2  SBS3  ...
    TCGA-A8-A06X-01  100.5  23.2  0.0  ...

The first column can be named ``Sample``, ``sample``, ``patient``,
``Tumor_Sample_Barcode``, or ``bcr_patient_barcode``.  Signature columns
must start with ``SBS``, ``DBS``, or ``ID`` (COSMIC naming convention).

Recommended source: Alexandrov et al. 2020 Cell — TCGA SBS signature
activities (Synapse syn11738669, requires free registration).  Place the
downloaded file in data/cache/ as ``TCGA_SBS_signatures.tsv`` and register
it with the FileManager before running ``tcga-db load``.
"""

import pandas as pd

from ...config import get_logger
from ..base import BaseProcessor

logger = get_logger(__name__)

# Column names that may contain the sample/patient identifier
_SAMPLE_COL_CANDIDATES = {
    "Sample",
    "sample",
    "patient",
    "patient_id",
    "Tumor_Sample_Barcode",
    "bcr_patient_barcode",
    "SAMPLE_ID",
}

# Prefixes that identify signature columns
_SIG_PREFIXES = ("SBS", "DBS", "ID")


def _is_signature_col(col: str) -> bool:
    """Return True if *col* looks like a COSMIC signature column.

    Args:
        col: Column name string.

    Returns:
        ``True`` when the name starts with a known COSMIC prefix.
    """
    return any(col.startswith(p) for p in _SIG_PREFIXES)


class MutationalSignaturesProcessor(BaseProcessor):
    """Load COSMIC SBS signature exposures into mutational_signatures.

    The source file is a wide matrix (samples × signatures).  This
    processor melts it to a long format, normalises patient IDs, and
    filters to patients present in the patients table.

    Args:
        batch_size: INSERT batch size override.
    """

    def __init__(self, batch_size: int | None = None) -> None:
        super().__init__(
            dataset_name="mutational_signatures", batch_size=batch_size
        )

    def get_table_name(self) -> str:
        """Return target table name."""
        return "mutational_signatures"

    def validate_data(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, list[str]]:
        """Identify sample and signature columns; validate format.

        Args:
            df: Raw DataFrame from :meth:`load_file`.

        Returns:
            Tuple of ``(df, warnings)``.

        Raises:
            ValueError: If no sample column or no signature columns found.
        """
        warnings: list[str] = []

        # Find sample identifier column
        sample_col: str | None = None
        for candidate in _SAMPLE_COL_CANDIDATES:
            if candidate in df.columns:
                sample_col = candidate
                break
        if sample_col is None:
            # Fall back: first column
            sample_col = df.columns[0]
            warnings.append(
                f"No known sample column found; using first column "
                f"'{sample_col}' as sample identifier."
            )

        sig_cols = [c for c in df.columns if _is_signature_col(c)]
        if not sig_cols:
            raise ValueError(
                f"No signature columns found (expected names starting with "
                f"{_SIG_PREFIXES}).  Available columns: {list(df.columns)[:20]}"
            )

        self.logger.info(
            "Found %d signature columns, sample column: '%s'",
            len(sig_cols),
            sample_col,
        )
        warnings.append(f"Found {len(sig_cols)} signature columns")

        # Tag the sample column for transform_data
        df = df.rename(columns={sample_col: "_sample_col"})
        df = df.dropna(subset=["_sample_col"])

        return df, warnings

    def transform_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Melt wide signature matrix to long format.

        Steps:
        1. Extract patient_id (first 12 chars if TCGA barcode).
        2. Melt wide → long (patient_id, signature_id, exposure).
        3. Filter to patients present in the patients table.
        4. Drop rows where exposure is NaN or zero.

        Args:
            df: Validated DataFrame from :meth:`validate_data`.

        Returns:
            Long-format DataFrame with columns
            ``(patient_id, signature_id, exposure)``.
        """
        # Normalise patient_id from barcode
        raw_id = df["_sample_col"].astype(str)
        is_barcode = raw_id.str.startswith("TCGA-")
        patient_ids = raw_id.copy()
        patient_ids[is_barcode] = raw_id[is_barcode].str[:12]
        df["patient_id"] = patient_ids

        sig_cols = [c for c in df.columns if _is_signature_col(c)]

        # Melt to long format
        long_df = df[["patient_id", *sig_cols]].melt(
            id_vars="patient_id",
            value_vars=sig_cols,
            var_name="signature_id",
            value_name="exposure",
        )

        # Keep only non-zero exposures (sparse representation)
        long_df["exposure"] = pd.to_numeric(
            long_df["exposure"], errors="coerce"
        )
        long_df = long_df.dropna(subset=["exposure"])
        long_df = long_df[long_df["exposure"] > 0.0]

        # Filter to patients in the patients table
        patient_result = self.db_manager.execute(
            "SELECT patient_id FROM patients"
        )
        known_patients = {row[0] for row in patient_result.fetchall()}
        before = len(long_df)
        long_df = long_df[long_df["patient_id"].isin(known_patients)]
        skipped = before - len(long_df)
        if skipped > 0:
            self.logger.info(
                "Skipped %d rows for patients not in patients table", skipped
            )

        self.logger.info(
            "%d signature exposure rows for %d patients",
            len(long_df),
            long_df["patient_id"].nunique(),
        )

        return long_df[["patient_id", "signature_id", "exposure"]].reset_index(
            drop=True
        )
