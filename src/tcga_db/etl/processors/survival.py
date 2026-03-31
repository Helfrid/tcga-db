"""Survival processor — loads TCGA clinical survival endpoints."""

from pathlib import Path

import numpy as np
import pandas as pd

from ..base import BaseProcessor


class SurvivalProcessor(BaseProcessor):
    """Loads TCGA survival data from the Liu et al. 2018 curated endpoints.

    Source: TCGA_survival_data
    Target: survival table (~11K patients)

    Four endpoints: OS, DSS, DFI, PFI — each with time (days) and event.
    """

    def __init__(self, batch_size: int | None = None) -> None:
        super().__init__(dataset_name="survival", batch_size=batch_size)

    def get_table_name(self) -> str:
        return "survival"

    def load_file(self, file_path: Path) -> pd.DataFrame:
        """Load survival TSV."""
        return pd.read_csv(file_path, sep="\t", low_memory=False)

    def validate_data(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, list[str]]:
        """Validate survival data columns and structure."""
        warnings: list[str] = []

        # Identify sample/patient column
        sample_col = None
        for col in ["sample", "_PATIENT", "bcr_patient_barcode"]:
            if col in df.columns:
                sample_col = col
                break

        if sample_col is None:
            raise ValueError(
                "Cannot find sample/patient column in survival data. "
                f"Available columns: {list(df.columns)}"
            )

        df = df.rename(columns={sample_col: "_sample_col"})

        # Check for survival endpoint columns
        endpoint_cols = {
            "OS",
            "OS.time",
            "DSS",
            "DSS.time",
            "DFI",
            "DFI.time",
            "PFI",
            "PFI.time",
        }
        found = endpoint_cols & set(df.columns)
        missing = endpoint_cols - found
        if missing:
            warnings.append(f"Missing survival columns: {missing}")

        # Drop rows with no sample identifier
        df = df.dropna(subset=["_sample_col"])
        warnings.append(f"Loaded {len(df)} survival records")

        return df, warnings

    def transform_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform survival data to table schema."""
        # Extract patient_id (first 12 chars if barcode, or use as-is)
        patient_id = df["_sample_col"].astype(str)
        # If these look like full barcodes, truncate to patient portion
        is_barcode = patient_id.str.startswith("TCGA-")
        patient_id[is_barcode] = patient_id[is_barcode].str[:12]

        # Deduplicate to one row per patient (keep first)
        result = pd.DataFrame({"patient_id": patient_id})

        # Map survival endpoints
        for endpoint in ["os", "dss", "dfi", "pfi"]:
            time_col = f"{endpoint.upper()}.time"
            event_col = endpoint.upper()

            if time_col in df.columns:
                result[f"{endpoint}_time"] = pd.to_numeric(
                    df[time_col], errors="coerce"
                )
            else:
                result[f"{endpoint}_time"] = np.nan

            if event_col in df.columns:
                # Convert 0/1 to boolean, handle NaN
                events = pd.to_numeric(df[event_col], errors="coerce")
                result[f"{endpoint}_event"] = events.map(
                    {1.0: True, 0.0: False}
                )
            else:
                result[f"{endpoint}_event"] = None

        # Deduplicate on patient_id
        result = result.drop_duplicates(subset=["patient_id"], keep="first")

        # Replace NaN with None for SQL NULL
        result = result.where(result.notna(), other=None)

        return result
