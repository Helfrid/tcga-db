"""Sample processor — loads TCGA + GTEx sample annotations."""

from pathlib import Path

import pandas as pd

from ..base import BaseProcessor


class SampleProcessor(BaseProcessor):
    """Loads sample phenotype annotations from the Toil phenotype file.

    Source: TcgaTargetGTEX_phenotype.txt.gz
    Target: samples table (~19K rows for TCGA + GTEx)

    Handles TCGA barcode parsing (patient_id, sample_type_code,
    is_tumour) and GTEx samples (patient_id=NULL).
    TARGET samples are excluded.
    """

    def __init__(self, batch_size: int | None = None) -> None:
        super().__init__(dataset_name="phenotype", batch_size=batch_size)

    def get_table_name(self) -> str:
        return "samples"

    def load_file(self, file_path: Path) -> pd.DataFrame:
        """Load phenotype TSV (gzipped)."""
        return pd.read_csv(
            file_path, sep="\t", low_memory=False, encoding="latin-1"
        )

    def validate_data(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, list[str]]:
        """Filter out TARGET samples and validate structure."""
        warnings: list[str] = []

        sample_col = df.columns[0]
        df = df.rename(columns={sample_col: "sample_id"})

        initial = len(df)

        # Filter out TARGET samples
        if "_study" in df.columns:
            study_upper = df["_study"].str.upper()
            df = df[study_upper.isin(["TCGA", "GTEX"])].copy()
            excluded = initial - len(df)
            if excluded > 0:
                warnings.append(
                    f"Excluded {excluded} non-TCGA/GTEx samples (TARGET etc.)"
                )
        else:
            warnings.append("No _study column found")

        # Drop rows with missing sample_id
        df = df.dropna(subset=["sample_id"])

        warnings.append(
            f"Retained {len(df)} samples "
            f"(TCGA: {(df['_study'].str.upper() == 'TCGA').sum()}, "
            f"GTEx: {(df['_study'].str.upper() == 'GTEX').sum()})"
        )

        return df, warnings

    def transform_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform phenotype data to samples table schema."""
        study_upper = df["_study"].str.upper()
        is_tcga = study_upper == "TCGA"

        # Parse TCGA barcodes
        patient_id = pd.Series(None, index=df.index, dtype="object")
        patient_id[is_tcga] = df.loc[is_tcga, "sample_id"].str[:12]

        sample_type_code = pd.Series(None, index=df.index, dtype="object")
        # TCGA barcodes: chars 13:15 (0-indexed) are sample type
        tcga_ids = df.loc[is_tcga, "sample_id"]
        has_type_code = tcga_ids.str.len() >= 15
        valid_tcga = is_tcga.copy()
        valid_tcga[is_tcga] = has_type_code.values
        sample_type_code[valid_tcga] = df.loc[valid_tcga, "sample_id"].str[
            13:15
        ]

        # Compute is_tumour: TCGA codes 01-09 are tumour
        is_tumour = pd.Series(False, index=df.index)
        has_code = sample_type_code.notna()
        is_tumour[has_code] = (
            sample_type_code[has_code].astype(int).between(1, 9)
        )

        # Primary site
        primary_site = df.get("_primary_site", pd.Series(dtype="object"))
        primary_site_normalised = primary_site.str.strip().str.lower()

        # Sample type label
        sample_type = df.get("_sample_type", pd.Series(dtype="object"))

        # Detailed category
        detailed = df.get(
            "detailed_category",
            df.get("_primary_disease", pd.Series(dtype="object")),
        )

        return pd.DataFrame(
            {
                "sample_id": df["sample_id"],
                "patient_id": patient_id,
                "source_study": study_upper,
                "primary_site": primary_site,
                "primary_site_normalised": primary_site_normalised,
                "detailed_category": detailed,
                "sample_type": sample_type,
                "sample_type_code": sample_type_code,
                "is_tumour": is_tumour,
            }
        )
