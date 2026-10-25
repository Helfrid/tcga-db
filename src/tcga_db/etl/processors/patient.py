"""Patient processor — extracts TCGA patients from phenotype data."""

from pathlib import Path

import pandas as pd

from ..base import BaseProcessor

#: Toil phenotype disease name → TCGA study code.
#: The phenotype file carries long disease names only; ``tissue_mapping``
#: and the wider TCGA literature key on the four-letter study code, so the
#: two cannot be joined without this translation.
DISEASE_TO_STUDY_CODE: dict[str, str] = {
    "Acute Myeloid Leukemia": "LAML",
    "Adrenocortical Cancer": "ACC",
    "Bladder Urothelial Carcinoma": "BLCA",
    "Brain Lower Grade Glioma": "LGG",
    "Breast Invasive Carcinoma": "BRCA",
    "Cervical & Endocervical Cancer": "CESC",
    "Cholangiocarcinoma": "CHOL",
    "Colon Adenocarcinoma": "COAD",
    "Diffuse Large B-Cell Lymphoma": "DLBC",
    "Esophageal Carcinoma": "ESCA",
    "Glioblastoma Multiforme": "GBM",
    "Head & Neck Squamous Cell Carcinoma": "HNSC",
    "Kidney Chromophobe": "KICH",
    "Kidney Clear Cell Carcinoma": "KIRC",
    "Kidney Papillary Cell Carcinoma": "KIRP",
    "Liver Hepatocellular Carcinoma": "LIHC",
    "Lung Adenocarcinoma": "LUAD",
    "Lung Squamous Cell Carcinoma": "LUSC",
    "Mesothelioma": "MESO",
    "Ovarian Serous Cystadenocarcinoma": "OV",
    "Pancreatic Adenocarcinoma": "PAAD",
    "Pheochromocytoma & Paraganglioma": "PCPG",
    "Prostate Adenocarcinoma": "PRAD",
    "Rectum Adenocarcinoma": "READ",
    "Sarcoma": "SARC",
    "Skin Cutaneous Melanoma": "SKCM",
    "Stomach Adenocarcinoma": "STAD",
    "Testicular Germ Cell Tumor": "TGCT",
    "Thymoma": "THYM",
    "Thyroid Carcinoma": "THCA",
    "Uterine Carcinosarcoma": "UCS",
    "Uterine Corpus Endometrioid Carcinoma": "UCEC",
    "Uveal Melanoma": "UVM",
}

#: Column in the Toil phenotype file holding the disease name.
_DISEASE_COL = "primary disease or tissue"

#: Assigned where the phenotype file carries no disease name at all.
#: The only such TCGA barcode is a "Control Analyte" (sample type 20),
#: which is not a tumour donor but is still referenced by ``samples``,
#: so it is labelled rather than dropped.
_CONTROL_CODE = "CONTROL"


class PatientProcessor(BaseProcessor):
    """Extracts unique TCGA patients from the Toil phenotype file.

    Source: TcgaTargetGTEX_phenotype.txt.gz
    Target: patients table (~11K rows)

    TCGA barcodes encode patient ID in the first 12 characters
    (e.g. TCGA-A7-A13E). Multiple samples map to the same patient.

    ``cancer_type`` is the four-letter TCGA study code (BRCA, LUAD, ...)
    translated from the phenotype file's long disease name via
    :data:`DISEASE_TO_STUDY_CODE`, and ``project_id`` is ``TCGA-<code>``.
    """

    EXPECTED_SOURCE_COLUMNS = (_DISEASE_COL, "_study")

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
        """Extract unique patients from TCGA barcodes.

        Args:
            df: Validated phenotype rows, TCGA only.

        Returns:
            One row per patient with ``patient_id``, ``project_id`` and
            ``cancer_type`` (four-letter TCGA study code).

        Raises:
            ValueError: If any disease name has no study-code mapping.
        """
        df = df.assign(patient_id=df["sample_id"].str[:12])

        patient_df = (
            df.groupby("patient_id")
            .agg(primary_disease=(_DISEASE_COL, "first"))
            .reset_index()
        )

        unmapped = sorted(
            set(patient_df["primary_disease"].dropna())
            - set(DISEASE_TO_STUDY_CODE)
        )
        if unmapped:
            raise ValueError(
                "No TCGA study code mapped for disease name(s): "
                f"{unmapped}. Add them to DISEASE_TO_STUDY_CODE."
            )

        cancer_type = (
            patient_df["primary_disease"]
            .map(DISEASE_TO_STUDY_CODE)
            .fillna(_CONTROL_CODE)
        )
        n_control = int((cancer_type == _CONTROL_CODE).sum())
        if n_control:
            self.logger.warning(
                "%d patient(s) have no disease name; labelled %s",
                n_control,
                _CONTROL_CODE,
            )
        patient_df = patient_df.assign(
            cancer_type=cancer_type,
            project_id="TCGA-" + cancer_type,
        )

        return patient_df[["patient_id", "project_id", "cancer_type"]]
