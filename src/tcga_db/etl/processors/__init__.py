"""ETL processors for TCGA/GTEx data loading."""

from .copy_number import CopyNumberWideProcessor
from .gene import GeneProcessor
from .mutational_signatures import MutationalSignaturesProcessor
from .patient import PatientProcessor
from .sample import SampleProcessor
from .sample_qc import SampleQCProcessor
from .survival import SurvivalProcessor
from .tissue_mapping import TissueMappingProcessor

__all__ = [
    "CopyNumberWideProcessor",
    "GeneProcessor",
    "MutationalSignaturesProcessor",
    "PatientProcessor",
    "SampleProcessor",
    "SampleQCProcessor",
    "SurvivalProcessor",
    "TissueMappingProcessor",
]
