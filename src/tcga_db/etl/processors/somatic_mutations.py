"""Somatic mutation processor — MC3 MAF from Xena Pan-Cancer Atlas.

Loads the unified MC3 somatic mutation call set into the somatic_mutations
table.  Each row is a single mutation event (SNV, indel) with functional
annotation.

Source: mc3.v0.2.8.PUBLIC.xena.gz
Target: somatic_mutations table (~3-4M rows)

The Xena MC3 file is a simplified MAF-like TSV.  Column names vary slightly
from the full MAF spec, so a flexible rename map is applied before
validation.
"""

import hashlib
from pathlib import Path

import pandas as pd

from ...config import get_logger
from ..base import BaseProcessor

logger = get_logger(__name__)

# Flexible column rename map: Xena / MAF variants → our schema names
_COL_MAP: dict[str, str] = {
    # Sample barcode
    "sample": "sample_id",
    "Sample": "sample_id",
    "Tumor_Sample_Barcode": "sample_id",
    # Chromosome
    "chr": "chromosome",
    "Chromosome": "chromosome",
    # Positions
    "start": "start_position",
    "Start_Position": "start_position",
    "Start_position": "start_position",
    "end": "end_position",
    "End_Position": "end_position",
    "End_position": "end_position",
    # Alleles
    "reference": "reference_allele",
    "Reference_Allele": "reference_allele",
    "alt": "tumor_allele",
    "Tumor_Seq_Allele2": "tumor_allele",
    # Gene
    "gene": "gene_symbol",
    "Hugo_Symbol": "gene_symbol",
    # Classification / type
    "Variant_Classification": "variant_classification",
    "Variant_Type": "variant_type",
    # Protein change
    "Amino_Acid_Change": "hgvsp_short",
    "HGVSp_Short": "hgvsp_short",
    "protein_change": "hgvsp_short",
    # VAF
    "DNA_VAF": "vaf",
    "t_vaf": "vaf",
    "tumor_f": "vaf",
}

_OUTPUT_COLS = [
    "mutation_id",
    "sample_id",
    "patient_id",
    "gene_id",
    "gene_symbol",
    "chromosome",
    "start_position",
    "end_position",
    "variant_classification",
    "variant_type",
    "reference_allele",
    "tumor_allele",
    "hgvsp_short",
    "vaf",
]

_OPTIONAL_COLS = _OUTPUT_COLS[5:]  # everything after gene_symbol


def _mutation_id(row_key: str) -> str:
    """Return a deterministic hex digest for a mutation key string.

    Args:
        row_key: Concatenated key fields separated by ``:``.

    Returns:
        32-character MD5 hex digest (not used for security).
    """
    return hashlib.md5(  # noqa: S324
        row_key.encode(), usedforsecurity=False
    ).hexdigest()


class SomaticMutationProcessor(BaseProcessor):
    """Load MC3 somatic mutations from the Xena Pan-Cancer Atlas.

    Applies flexible column remapping, filters to TCGA samples, and
    performs a gene-symbol lookup to populate ``gene_id`` FK.

    Args:
        batch_size: INSERT batch size override.
    """

    def __init__(self, batch_size: int | None = None) -> None:
        super().__init__(
            dataset_name="somatic_mutations", batch_size=batch_size
        )

    def get_table_name(self) -> str:
        """Return target table name."""
        return "somatic_mutations"

    def load_file(self, file_path: Path) -> pd.DataFrame:
        """Load MAF TSV (latin-1 to handle non-ASCII characters).

        Args:
            file_path: Path to the TSV or TSV.gz file.

        Returns:
            Raw DataFrame.
        """
        return pd.read_csv(
            file_path,
            sep="\t",
            low_memory=False,
            encoding="latin-1",
        )

    def validate_data(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, list[str]]:
        """Rename columns and check for required fields.

        Args:
            df: Raw DataFrame from :meth:`load_file`.

        Returns:
            Tuple of ``(renamed_df, warnings)``.

        Raises:
            ValueError: If sample_id or gene_symbol columns are absent.
        """
        warnings: list[str] = []

        rename = {c: _COL_MAP[c] for c in df.columns if c in _COL_MAP}
        df = df.rename(columns=rename)

        required = {"sample_id", "gene_symbol"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"Required columns missing from mutations file: {missing}. "
                f"Available columns: {list(df.columns)[:20]}"
            )

        before = len(df)
        df = df.dropna(subset=["sample_id", "gene_symbol"])
        dropped = before - len(df)
        if dropped:
            warnings.append(
                f"Dropped {dropped} rows with missing sample or gene"
            )

        warnings.append(f"Loaded {len(df)} raw mutation records")
        return df, warnings

    def transform_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform to somatic_mutations schema.

        Steps:
        1. Filter to TCGA barcodes.
        2. Extract patient_id (first 12 characters of barcode).
        3. Build deterministic mutation_id from key fields.
        4. Look up gene_id by gene symbol from the genes table.
        5. Fill optional columns with None when absent.
        6. Convert numeric columns.

        Args:
            df: Validated DataFrame from :meth:`validate_data`.

        Returns:
            DataFrame ready for batch insertion.
        """
        # Filter to TCGA barcodes only
        mask = df["sample_id"].astype(str).str.startswith("TCGA-")
        df = df[mask].copy()
        self.logger.info(
            "Kept %d TCGA mutations (of %d total)", len(df), mask.sum()
        )

        # Filter to samples present in the samples table (MC3 includes
        # samples without RNA-seq that are absent from our samples table)
        sample_result = self.db_manager.execute(
            "SELECT sample_id FROM samples"
        )
        known_samples = {row[0] for row in sample_result.fetchall()}
        before = len(df)
        df = df[df["sample_id"].isin(known_samples)]
        skipped = before - len(df)
        if skipped:
            self.logger.info(
                "Skipped %d mutations for samples not in samples table",
                skipped,
            )

        # Extract patient_id from first 12 chars of barcode
        df["patient_id"] = df["sample_id"].astype(str).str[:12]

        # Deterministic mutation_id
        chrom = df.get("chromosome", pd.Series([""] * len(df), index=df.index))
        start = df.get(
            "start_position", pd.Series([""] * len(df), index=df.index)
        )
        ref = df.get(
            "reference_allele", pd.Series([""] * len(df), index=df.index)
        )
        alt = df.get("tumor_allele", pd.Series([""] * len(df), index=df.index))
        keys = (
            df["sample_id"].astype(str)
            + ":"
            + chrom.astype(str)
            + ":"
            + start.astype(str)
            + ":"
            + ref.astype(str)
            + ":"
            + alt.astype(str)
        )
        df["mutation_id"] = keys.map(_mutation_id)

        # Gene symbol → gene_id FK lookup
        gene_result = self.db_manager.execute(
            "SELECT gene_symbol, gene_id FROM genes "
            "WHERE is_protein_coding = TRUE"
        )
        gene_map: dict[str, str] = {
            row[0]: row[1] for row in gene_result.fetchall()
        }
        df["gene_id"] = df["gene_symbol"].map(gene_map)

        # Add any optional columns that are absent
        for col in _OPTIONAL_COLS:
            if col not in df.columns:
                df[col] = None

        result = df[_OUTPUT_COLS].copy()

        # Numeric coercions
        for col in ("start_position", "end_position"):
            result[col] = pd.to_numeric(result[col], errors="coerce")
        result["vaf"] = pd.to_numeric(result["vaf"], errors="coerce")

        result = result.where(result.notna(), other=None)
        return result
