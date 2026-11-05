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
    # Classification / type.  The Xena MC3 file calls this "effect" and
    # carries the standard MAF vocabulary (Missense_Mutation, Silent, ...).
    # Without this entry variant_classification was silently NULL for
    # every row — see _OPTIONAL_COLS backfill below.
    "effect": "variant_classification",
    "Variant_Classification": "variant_classification",
    "Variant_Type": "variant_type",
    # Protein change
    "Amino_Acid_Change": "hgvsp_short",
    "HGVSp_Short": "hgvsp_short",
    "protein_change": "hgvsp_short",
    # Functional prediction (present in the Xena MC3 file)
    "SIFT": "sift",
    "PolyPhen": "polyphen",
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
    "sift",
    "polyphen",
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


def _derive_variant_type(ref: pd.Series, alt: pd.Series) -> pd.Series:
    """Infer MAF ``Variant_Type`` from reference and tumour alleles.

    The Xena MC3 file omits the column, but the standard categories
    follow directly from the allele strings: a ``-`` placeholder marks an
    insertion or deletion, and equal-length substitutions are SNP, DNP,
    TNP or ONP by length.

    Args:
        ref: Reference alleles.
        alt: Tumour alleles.

    Returns:
        Series of variant-type strings, ``None`` where undetermined.
    """
    ref_s = ref.fillna("").astype(str)
    alt_s = alt.fillna("").astype(str)
    len_by_n = {1: "SNP", 2: "DNP", 3: "TNP"}

    def classify(r: str, a: str) -> str | None:
        if not r or not a:
            return None
        if r == "-" or len(a) > len(r):
            return "INS"
        if a == "-" or len(r) > len(a):
            return "DEL"
        return len_by_n.get(len(r), "ONP")

    return pd.Series(
        [classify(r, a) for r, a in zip(ref_s, alt_s, strict=True)],
        index=ref.index,
        dtype=object,
    )


class SomaticMutationProcessor(BaseProcessor):
    """Load MC3 somatic mutations from the Xena Pan-Cancer Atlas.

    Applies flexible column remapping, filters to TCGA samples, and
    performs a gene-symbol lookup to populate ``gene_id`` FK.

    Args:
        batch_size: INSERT batch size override.
    """

    EXPECTED_SOURCE_COLUMNS = ("sample", "gene", "effect")

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

    def clear_table(self) -> None:
        """Drop and recreate the table instead of deleting its rows.

        ``DELETE FROM somatic_mutations`` crashes DuckDB 1.5.1 with
        SIGTRAP at this scale (2.3 M rows, three foreign keys); the
        transaction rolls back cleanly but the process dies, so a
        force-reload can never complete.  Dropping and recreating from
        the schema definition is equivalent here — the table is always
        repopulated in full from the MC3 file — and avoids the crash.
        """
        from ...database.schema import Schema, apply_column_migrations

        table_name = self.get_table_name()
        self.logger.warning("Dropping and recreating table: %s", table_name)
        self.db_manager.execute(f"DROP TABLE IF EXISTS {table_name}")
        Schema().create_table(table_name)
        apply_column_migrations()

    def insert_batch(self, df: pd.DataFrame) -> tuple[int, int]:
        """Bulk-insert a batch via DuckDB DataFrame registration.

        The inherited row-by-row ``executemany`` path crashes DuckDB
        (SIGTRAP) on this table at full scale — 2.3 M rows across 16
        columns with three foreign keys.  Registering the frame and
        running a single ``INSERT ... SELECT`` avoids that, and is an
        order of magnitude faster besides.

        Args:
            df: DataFrame whose columns map directly to table columns.

        Returns:
            A two-tuple of ``(inserted, skipped)``.
        """
        if df.empty:
            return 0, 0

        con = self.db_manager.connect()
        col_list = ", ".join(df.columns)
        con.register("_mutation_batch", df)
        try:
            con.execute(
                f"INSERT INTO {self.get_table_name()} ({col_list}) "
                f"SELECT {col_list} FROM _mutation_batch"  # noqa: S608
            )
        finally:
            con.unregister("_mutation_batch")

        return len(df), 0

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

        # variant_type is absent from the Xena MC3 file but is
        # recoverable from the allele lengths.
        if "variant_type" not in df.columns:
            df["variant_type"] = _derive_variant_type(
                df["reference_allele"], df["tumor_allele"]
            )

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
