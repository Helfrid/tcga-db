"""ETL pipeline abstractions for tcga-db data ingestion."""

from .base import BaseProcessor, ProcessingResult

__all__ = [
    "BaseProcessor",
    "ProcessingResult",
]
