"""Tissue mapping processor — seeds curated TCGA↔GTEx tissue mappings."""

from pathlib import Path

import pandas as pd

from ..base import BaseProcessor, ProcessingResult

# Curated mapping of TCGA cancer types to matching GTEx normal tissues.
# Based on anatomical correspondence from the literature.
TCGA_GTEX_TISSUE_MAP: list[dict[str, str]] = [
    {
        "tcga_cancer_type": "BRCA",
        "gtex_tissue": "Breast - Mammary Tissue",
        "primary_site_normalised": "breast",
    },
    {
        "tcga_cancer_type": "LUAD",
        "gtex_tissue": "Lung",
        "primary_site_normalised": "lung",
    },
    {
        "tcga_cancer_type": "LUSC",
        "gtex_tissue": "Lung",
        "primary_site_normalised": "lung",
    },
    {
        "tcga_cancer_type": "PRAD",
        "gtex_tissue": "Prostate",
        "primary_site_normalised": "prostate",
    },
    {
        "tcga_cancer_type": "THCA",
        "gtex_tissue": "Thyroid",
        "primary_site_normalised": "thyroid",
    },
    {
        "tcga_cancer_type": "COAD",
        "gtex_tissue": "Colon - Sigmoid",
        "primary_site_normalised": "colon",
    },
    {
        "tcga_cancer_type": "COAD",
        "gtex_tissue": "Colon - Transverse",
        "primary_site_normalised": "colon",
    },
    {
        "tcga_cancer_type": "READ",
        "gtex_tissue": "Colon - Sigmoid",
        "primary_site_normalised": "colon",
    },
    {
        "tcga_cancer_type": "READ",
        "gtex_tissue": "Colon - Transverse",
        "primary_site_normalised": "colon",
    },
    {
        "tcga_cancer_type": "STAD",
        "gtex_tissue": "Stomach",
        "primary_site_normalised": "stomach",
    },
    {
        "tcga_cancer_type": "LIHC",
        "gtex_tissue": "Liver",
        "primary_site_normalised": "liver",
    },
    {
        "tcga_cancer_type": "KIRC",
        "gtex_tissue": "Kidney - Cortex",
        "primary_site_normalised": "kidney",
    },
    {
        "tcga_cancer_type": "KIRP",
        "gtex_tissue": "Kidney - Cortex",
        "primary_site_normalised": "kidney",
    },
    {
        "tcga_cancer_type": "KICH",
        "gtex_tissue": "Kidney - Cortex",
        "primary_site_normalised": "kidney",
    },
    {
        "tcga_cancer_type": "UCEC",
        "gtex_tissue": "Uterus",
        "primary_site_normalised": "uterus",
    },
    {
        "tcga_cancer_type": "CESC",
        "gtex_tissue": "Cervix - Ectocervix",
        "primary_site_normalised": "cervix",
    },
    {
        "tcga_cancer_type": "CESC",
        "gtex_tissue": "Cervix - Endocervix",
        "primary_site_normalised": "cervix",
    },
    {
        "tcga_cancer_type": "BLCA",
        "gtex_tissue": "Bladder",
        "primary_site_normalised": "bladder",
    },
    {
        "tcga_cancer_type": "ESCA",
        "gtex_tissue": "Esophagus - Mucosa",
        "primary_site_normalised": "esophagus",
    },
    {
        "tcga_cancer_type": "ESCA",
        "gtex_tissue": "Esophagus - Muscularis",
        "primary_site_normalised": "esophagus",
    },
    {
        "tcga_cancer_type": "PAAD",
        "gtex_tissue": "Pancreas",
        "primary_site_normalised": "pancreas",
    },
    {
        "tcga_cancer_type": "SKCM",
        "gtex_tissue": "Skin - Not Sun Exposed (Suprapubic)",
        "primary_site_normalised": "skin",
    },
    {
        "tcga_cancer_type": "SKCM",
        "gtex_tissue": "Skin - Sun Exposed (Lower leg)",
        "primary_site_normalised": "skin",
    },
    {
        "tcga_cancer_type": "OV",
        "gtex_tissue": "Ovary",
        "primary_site_normalised": "ovary",
    },
]


class TissueMappingProcessor(BaseProcessor):
    """Seeds the tissue_mapping table from curated static data.

    No file download required — data is defined in this module.
    Target: tissue_mapping table (~25 rows)
    """

    def __init__(self, batch_size: int | None = None) -> None:
        super().__init__(dataset_name="tissue_mapping", batch_size=batch_size)

    def get_table_name(self) -> str:
        return "tissue_mapping"

    def validate_data(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, list[str]]:
        """Validate tissue mapping data."""
        warnings: list[str] = []
        required = {
            "tcga_cancer_type",
            "gtex_tissue",
            "primary_site_normalised",
        }
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        warnings.append(f"Loaded {len(df)} tissue mappings")
        return df, warnings

    def transform_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """No transformation needed — data is already in schema format."""
        return df

    def process_file(
        self, file_path: Path | None = None, force_reload: bool = False
    ) -> ProcessingResult:
        """Load tissue mappings from static data (ignores file_path).

        Args:
            file_path: Ignored — data comes from TCGA_GTEX_TISSUE_MAP.
            force_reload: If True, clear and reload.
        """
        import time

        from ..base import ProcessingResult

        start = time.time()

        existing = self.get_existing_record_count()
        if existing > 0 and not force_reload:
            return ProcessingResult(
                processor_name=self.__class__.__name__,
                dataset_name=self.dataset_name,
                source_file=Path("static"),
                records_processed=0,
                records_inserted=0,
                records_skipped=existing,
                processing_time_seconds=time.time() - start,
                status="skipped",
            )

        if force_reload:
            self.clear_table()

        df = pd.DataFrame(TCGA_GTEX_TISSUE_MAP)
        df, warnings = self.validate_data(df)
        df = self.transform_data(df)

        inserted, _ = self.insert_batch(df)

        return ProcessingResult(
            processor_name=self.__class__.__name__,
            dataset_name=self.dataset_name,
            source_file=Path("static"),
            records_processed=len(df),
            records_inserted=inserted,
            records_skipped=0,
            processing_time_seconds=time.time() - start,
            status="success",
            warnings=warnings,
        )
