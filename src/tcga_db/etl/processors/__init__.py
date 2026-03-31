"""ETL processors for TCGA/GTEx data loading."""

from .gene import GeneProcessor
from .patient import PatientProcessor
from .sample import SampleProcessor
from .survival import SurvivalProcessor
from .tissue_mapping import TissueMappingProcessor

__all__ = [
    "GeneProcessor",
    "PatientProcessor",
    "SampleProcessor",
    "SurvivalProcessor",
    "TissueMappingProcessor",
]
