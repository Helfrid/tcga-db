"""Gene metadata processor for GENCODE v23 data.

Combines the Xena probemap (gene coordinates, symbols) with the
GENCODE v23 GTF annotation (gene biotype/type) to produce a complete
gene metadata table.
"""

import gzip
from pathlib import Path

import pandas as pd

from ..base import BaseProcessor


def _parse_gene_types_from_gtf(gtf_path: Path) -> dict[str, str]:
    """Extract gene_id → gene_type mapping from a GENCODE GTF.

    Only parses 'gene' feature lines. Returns a dict mapping
    versioned Ensembl IDs (e.g. ENSG00000223972.5) to their
    gene_type (e.g. protein_coding, lincRNA, pseudogene).

    Args:
        gtf_path: Path to the (optionally gzipped) GTF file.
    """
    gene_types: dict[str, str] = {}

    opener = gzip.open if str(gtf_path).endswith(".gz") else open
    with opener(gtf_path, "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            fields = line.split("\t")
            if len(fields) < 9 or fields[2] != "gene":
                continue
            attrs = fields[8]
            gene_id = _extract_attr(attrs, "gene_id")
            gene_type = _extract_attr(attrs, "gene_type")
            if gene_id and gene_type:
                gene_types[gene_id] = gene_type

    return gene_types


def _extract_attr(attr_string: str, key: str) -> str | None:
    """Extract a value from a GTF attribute string."""
    for part in attr_string.split(";"):
        part = part.strip()
        if part.startswith(f'{key} "'):
            return part.split('"')[1]
    return None


class GeneProcessor(BaseProcessor):
    """Loads gene metadata from probemap + GENCODE v23 GTF.

    Source: gencode.v23.annotation.gene.probemap (coordinates/symbols)
            gencode.v23.annotation.gtf.gz (gene biotypes)
    Target: genes table (~60K rows)
    """

    def __init__(
        self,
        gtf_path: Path | None = None,
        batch_size: int | None = None,
    ) -> None:
        super().__init__(dataset_name="gene_probemap", batch_size=batch_size)
        self.gtf_path = gtf_path

    def get_table_name(self) -> str:
        return "genes"

    def load_file(self, file_path: Path) -> pd.DataFrame:
        """Load probemap TSV file."""
        return pd.read_csv(file_path, sep="\t", low_memory=False)

    def validate_data(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, list[str]]:
        """Validate probemap data and merge gene_type from GTF."""
        warnings: list[str] = []
        required = {"id", "gene", "chrom", "chromStart", "chromEnd", "strand"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        initial = len(df)

        # Drop rows with missing gene ID or symbol
        df = df.dropna(subset=["id", "gene"])
        if len(df) < initial:
            dropped = initial - len(df)
            warnings.append(f"Dropped {dropped} rows with missing id/gene")

        # Deduplicate on id (versioned Ensembl ID)
        dupes = df["id"].duplicated()
        if dupes.any():
            n = dupes.sum()
            warnings.append(f"Removed {n} duplicate Ensembl IDs")
            df = df[~dupes]

        # Get gene_type from GTF if available
        if self.gtf_path and self.gtf_path.exists():
            self.logger.info("Parsing gene types from GTF: %s", self.gtf_path)
            gene_types = _parse_gene_types_from_gtf(self.gtf_path)
            df = df.assign(
                gene_type=df["id"].map(gene_types).fillna("unknown")
            )
            n_typed = (df["gene_type"] != "unknown").sum()
            warnings.append(
                f"Mapped gene_type for {n_typed}/{len(df)} genes from GTF"
            )
        elif "gene_type" not in df.columns:
            warnings.append(
                "No GTF provided and no gene_type column — "
                "setting to 'unknown'"
            )
            df = df.assign(gene_type="unknown")

        return df, warnings

    def transform_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform probemap data to genes table schema."""
        return pd.DataFrame(
            {
                "gene_id": df["id"].str.split(".").str[0],
                "ensembl_id_versioned": df["id"],
                "gene_symbol": df["gene"],
                "gene_type": df["gene_type"],
                "chromosome": df["chrom"],
                "chrom_start": pd.to_numeric(
                    df["chromStart"], errors="coerce"
                ).astype("Int64"),
                "chrom_end": pd.to_numeric(
                    df["chromEnd"], errors="coerce"
                ).astype("Int64"),
                "strand": df["strand"],
                "is_protein_coding": df["gene_type"] == "protein_coding",
            }
        )
