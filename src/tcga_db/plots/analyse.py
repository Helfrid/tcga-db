"""Top-level single-call API for gene expression + survival analysis.

Example:
    >>> from tcga_db.plots import analyse_gene
    >>> results = analyse_gene("BRCA1", db_path="data/tcga.duckdb")
    >>> results["expression_fig"].show()
    >>> results["de_results"].head()
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt

from .data import get_gene_expression, get_survival_data, resolve_gene
from .expression import plot_pan_cancer_expression
from .stats import run_differential_expression, run_survival_analysis
from .survival import plot_survival_panels

if TYPE_CHECKING:
    pass

# Endpoint labels for display
_ENDPOINT_LABELS: dict[str, str] = {
    "os": "Overall Survival",
    "dss": "Disease-Specific Survival",
    "dfi": "Disease-Free Interval",
    "pfi": "Progression-Free Interval",
}


def analyse_gene(
    gene: str,
    db_path: str | Path = "data/tcga.duckdb",
    cancer_types: list[str] | None = None,
    survival_endpoint: str = "os",
    output_dir: str | Path | None = None,
    save_formats: list[str] | None = None,
    expression_figsize: tuple[float, float] = (22, 12),
    survival_figsize: tuple[float, float] = (22, 12),
    show: bool = False,
) -> dict:
    """Full expression + survival analysis for a single gene.

    Orchestrates all three layers:

    1. Fetches expression data from DuckDB.
    2. Runs Wilcoxon differential expression per cancer type.
    3. Generates the pan-cancer violin + jitter figure.
    4. Fetches survival data and runs log-rank tests.
    5. Generates Kaplan-Meier survival panels.

    Args:
        gene: Hugo gene symbol (e.g. ``"BRCA1"``) or unversioned Ensembl
            ID (e.g. ``"ENSG00000012048"``).
        db_path: Path to the ``tcga.duckdb`` file.
        cancer_types: Optional list of TCGA cancer type codes to restrict
            analysis.  ``None`` includes all types with GTEx-matched tissue.
        survival_endpoint: One of ``"os"``, ``"dss"``, ``"dfi"``,
            ``"pfi"``.  Defaults to ``"os"`` (Overall Survival).
        output_dir: If given, figures are saved here.  The directory is
            created if it does not exist.
        save_formats: List of file extensions to save
            (e.g. ``["png", "pdf"]``).  Requires ``output_dir``.
            Defaults to ``["png"]`` when ``output_dir`` is set.
        expression_figsize: ``(width, height)`` for the expression figure.
        survival_figsize: ``(width, height)`` for the survival figure.
        show: If ``True``, call ``plt.show()`` after generating figures.

    Returns:
        Dictionary with keys:

        * ``"gene_id"`` — resolved Ensembl ID
        * ``"gene_symbol"`` — resolved Hugo symbol
        * ``"expression_df"`` — raw expression DataFrame
        * ``"de_results"`` — per-cancer-type DE stats DataFrame
        * ``"survival_df"`` — survival DataFrame
        * ``"survival_results"`` — per-cancer-type survival stats DataFrame
        * ``"expression_fig"`` — :class:`matplotlib.figure.Figure`
        * ``"survival_fig"`` — :class:`matplotlib.figure.Figure`

    Raises:
        ValueError: If the gene is not found or the endpoint is invalid.
    """
    db_path = Path(db_path)
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    # ── Resolve gene ───────────────────────────────────────────────────
    gene_id, gene_symbol = resolve_gene(gene, db_path)

    # ── Expression analysis ────────────────────────────────────────────
    expression_df = get_gene_expression(
        gene_id,
        db_path,
        expression_type="tpm",
        cancer_types=cancer_types,
    )

    de_results = run_differential_expression(expression_df)

    expression_fig = plot_pan_cancer_expression(
        expression_df=expression_df,
        stats_df=de_results,
        gene_symbol=gene_symbol,
        figsize=expression_figsize,
    )

    # ── Survival analysis ──────────────────────────────────────────────
    survival_df = get_survival_data(
        gene_id,
        db_path,
        cancer_types=cancer_types,
        endpoint=survival_endpoint,
    )

    survival_results = run_survival_analysis(survival_df)

    endpoint_label = _ENDPOINT_LABELS.get(
        survival_endpoint, survival_endpoint.upper()
    )
    survival_fig = plot_survival_panels(
        survival_df=survival_df,
        survival_stats=survival_results,
        gene_symbol=gene_symbol,
        endpoint_label=endpoint_label,
        figsize=survival_figsize,
    )

    # ── Save figures ───────────────────────────────────────────────────
    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        formats = save_formats or ["png"]
        for fmt in formats:
            expression_fig.savefig(
                out / f"{gene_symbol}_expression.{fmt}",
                bbox_inches="tight",
                dpi=150,
            )
            survival_fig.savefig(
                out / f"{gene_symbol}_{survival_endpoint}_survival.{fmt}",
                bbox_inches="tight",
                dpi=150,
            )

    if show:
        plt.show()

    return {
        "gene_id": gene_id,
        "gene_symbol": gene_symbol,
        "expression_df": expression_df,
        "de_results": de_results,
        "survival_df": survival_df,
        "survival_results": survival_results,
        "expression_fig": expression_fig,
        "survival_fig": survival_fig,
    }
