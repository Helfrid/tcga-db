"""Sample QC processor — tumour purity, ploidy, and MSI status.

Loads ABSOLUTE/ASCAT purity and ploidy estimates plus MSI status into the
sample_qc table.

Primary source: TCGA_mastercalls.abs_tables_JSedit.fixed.txt
  (Xena Pan-Cancer Atlas)

Expected columns (flexible rename map handles ABSOLUTE output variants):

    Tumor_Sample_Barcode  Purity  Ploidy  Genome.doublings  ...

MSI status is optional; if not present in the ABSOLUTE file it can be
loaded from a separate clinical annotation file by running the processor
a second time with ``force_reload=False`` (rows are REPLACED, so existing
purity data is preserved only if both files have the same sample_id).

If MSI status is absent from the file, the column is populated with NULL.
"""

import pandas as pd

from ...config import get_logger
from ..base import BaseProcessor

logger = get_logger(__name__)

# Flexible column rename: source variants → our schema
_COL_MAP: dict[str, str] = {
    # Sample barcode
    "array": "sample_id",
    "Tumor_Sample_Barcode": "sample_id",
    "sample": "sample_id",
    "Sample": "sample_id",
    "sampleID": "sample_id",
    # Purity
    "Purity": "tumour_purity",
    "purity": "tumour_purity",
    "Cancer.DNA.fraction": "tumour_purity",
    # Ploidy
    "Ploidy": "ploidy",
    "ploidy": "ploidy",
    # Genome doublings (WGD indicator)
    "Genome.doublings": "genome_doublings",
    "genome_doublings": "genome_doublings",
    "WGD": "genome_doublings",
    # MSI status (optional)
    "MSI_status": "msi_status",
    "MSI.status": "msi_status",
    "msi_status": "msi_status",
    "MANTIS_MSI_status": "msi_status",
}

_OUTPUT_COLS = [
    "sample_id",
    "tumour_purity",
    "ploidy",
    "genome_doublings",
    "msi_status",
]


class SampleQCProcessor(BaseProcessor):
    """Load ABSOLUTE tumour purity/ploidy and MSI status into sample_qc.

    Filters to samples present in the samples table.  Numeric columns
    are coerced; genome_doublings is cast to integer; msi_status is
    stored as-is (e.g. ``"MSI-H"``, ``"MSS"``).

    Args:
        batch_size: INSERT batch size override.
    """

    def __init__(self, batch_size: int | None = None) -> None:
        super().__init__(dataset_name="sample_qc", batch_size=batch_size)

    def get_table_name(self) -> str:
        """Return target table name."""
        return "sample_qc"

    def validate_data(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, list[str]]:
        """Rename columns and check for required sample_id.

        Args:
            df: Raw DataFrame from :meth:`load_file`.

        Returns:
            Tuple of ``(renamed_df, warnings)``.

        Raises:
            ValueError: If no sample barcode column can be identified.
        """
        warnings: list[str] = []

        rename = {c: _COL_MAP[c] for c in df.columns if c in _COL_MAP}
        df = df.rename(columns=rename)

        if "sample_id" not in df.columns:
            raise ValueError(
                "Cannot identify sample barcode column in QC file. "
                f"Available columns: {list(df.columns)[:20]}"
            )

        before = len(df)
        df = df.dropna(subset=["sample_id"])
        dropped = before - len(df)
        if dropped:
            warnings.append(f"Dropped {dropped} rows with missing sample_id")

        # Filter to TCGA barcodes
        mask = df["sample_id"].astype(str).str.startswith("TCGA-")
        non_tcga = (~mask).sum()
        if non_tcga:
            warnings.append(f"Dropped {non_tcga} non-TCGA samples")
        df = df[mask]

        warnings.append(f"Loaded {len(df)} sample QC records")
        return df, warnings

    def transform_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform to sample_qc schema.

        Steps:
        1. Fill optional columns absent from the source file with None.
        2. Coerce numeric columns; cast genome_doublings to int.
        3. Filter to samples present in the samples table.
        4. Replace NaN with None for SQL NULL.

        Args:
            df: Validated DataFrame from :meth:`validate_data`.

        Returns:
            DataFrame ready for batch insertion.
        """
        # Add optional columns if absent
        for col in (
            "tumour_purity",
            "ploidy",
            "genome_doublings",
            "msi_status",
        ):
            if col not in df.columns:
                df[col] = None

        result = df[_OUTPUT_COLS].copy()

        # Numeric coercions
        for col in ("tumour_purity", "ploidy"):
            result[col] = pd.to_numeric(result[col], errors="coerce")

        # genome_doublings: coerce then cast to nullable int
        result["genome_doublings"] = pd.to_numeric(
            result["genome_doublings"], errors="coerce"
        )
        result["genome_doublings"] = result["genome_doublings"].where(
            result["genome_doublings"].notna(), other=None
        )

        # Filter to known samples
        sample_result = self.db_manager.execute(
            "SELECT sample_id FROM samples"
        )
        known_samples = {row[0] for row in sample_result.fetchall()}
        before = len(result)
        result = result[result["sample_id"].isin(known_samples)]
        skipped = before - len(result)
        if skipped > 0:
            self.logger.info(
                "Skipped %d rows for samples not in samples table", skipped
            )

        # Deduplicate on sample_id
        result = result.drop_duplicates(subset=["sample_id"], keep="first")

        result = result.where(result.notna(), other=None)
        return result
