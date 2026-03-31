"""Patient processor — extracts TCGA patients from phenotype data."""

from pathlib import Path

import pandas as pd

from ..base import BaseProcessor


class PatientProcessor(BaseProcessor):
    """Extracts unique TCGA patients from the Toil phenotype file.

    Source: TcgaTargetGTEX_phenotype.txt.gz
    Target: patients table (~11K rows)

    TCGA barcodes encode patient ID in the first 12 characters
    (e.g. TCGA-A7-A13E). Multiple samples map to the same patient.
    """

    def __init__(self, batch_size: int | None = None) -> None:
        super().__init__(dataset_name="phenotype", batch_size=batch_size)

    def get_table_name(self) -> str:
        return "patients"

    def load_file(self, file_path: Path) -> pd.DataFrame:
        """Load phenotype TSV (gzipped)."""
        return pd.read_csv(
            file_path, sep="\t", low_memory=False, encoding="latin-1"
        )

    def validate_data(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, list[str]]:
        """Filter to TCGA samples and validate barcode structure."""
        warnings: list[str] = []

        # Identify the sample ID column (first column)
        sample_col = df.columns[0]

        # Filter to TCGA only
        if "_study" in df.columns:
            tcga_mask = df["_study"].str.upper() == "TCGA"
            df = df[tcga_mask].copy()
            warnings.append(f"Filtered to {len(df)} TCGA samples")
        else:
            warnings.append("No _study column — assuming all rows are TCGA")

        # Validate barcode format (should start with TCGA-)
        valid_barcode = df[sample_col].str.startswith("TCGA-")
        invalid_count = (~valid_barcode).sum()
        if invalid_count > 0:
            warnings.append(
                f"Dropped {invalid_count} rows with invalid TCGA barcodes"
            )
            df = df[valid_barcode].copy()

        df = df.rename(columns={sample_col: "sample_id"})
        return df, warnings

    def transform_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract unique patients from TCGA barcodes."""
        # Extract patient_id (first 12 chars of barcode)
        df = df.assign(patient_id=df["sample_id"].str[:12])

        # Derive project_id and cancer_type
        # project_id comes from _primary_disease or the barcode TSS
        # For now derive from phenotype columns if available
        if "_primary_disease" in df.columns:
            patient_df = (
                df.groupby("patient_id")
                .agg(primary_disease=("_primary_disease", "first"))
                .reset_index()
            )
            # Map disease to TCGA project codes
            patient_df = patient_df.assign(
                project_id="TCGA-"
                + patient_df["primary_disease"].fillna("UNKNOWN"),
                cancer_type=patient_df["primary_disease"].fillna("UNKNOWN"),
            )
        else:
            # Fallback: extract from barcode TSS code
            patient_df = (
                df[["patient_id"]].drop_duplicates().reset_index(drop=True)
            )
            patient_df = patient_df.assign(
                project_id="TCGA-UNKNOWN",
                cancer_type="UNKNOWN",
            )

        return patient_df[["patient_id", "project_id", "cancer_type"]]
