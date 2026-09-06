"""tcga_db.plots — expression and survival analysis with publication figures.

Quick start::

    from tcga_db.plots import analyse_gene

    results = analyse_gene("TP53", db_path="data/tcga.duckdb")
    results["expression_fig"].savefig("TP53_expression.png", bbox_inches="tight")

Layers:

* :mod:`.data` — DuckDB query functions
* :mod:`.stats` — Wilcoxon DE + log-rank survival tests
* :mod:`.expression` — pan-cancer violin + jitter figure
* :mod:`.survival` — Kaplan-Meier survival panels
* :mod:`.analyse` — :func:`analyse_gene` top-level orchestrator
"""

from .analyse import analyse_gene
from .data import (
    get_gene_expression,
    get_survival_data,
    get_tissue_mapping,
    resolve_gene,
)
from .stats import run_differential_expression, run_survival_analysis

__all__ = [
    "analyse_gene",
    "get_gene_expression",
    "get_survival_data",
    "get_tissue_mapping",
    "resolve_gene",
    "run_differential_expression",
    "run_survival_analysis",
]
