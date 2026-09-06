"""Layer 3 — Kaplan-Meier survival plots.

Faceted KM curves stratified by high / low gene expression (median split),
one panel per cancer type.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from lifelines import KaplanMeierFitter

if TYPE_CHECKING:
    import matplotlib.figure


_COLOUR_HIGH = "#C0392B"  # red — high expression
_COLOUR_LOW = "#2980B9"  # blue — low expression
_ALPHA_CI = 0.15


def plot_survival_panels(
    survival_df: pd.DataFrame,
    survival_stats: pd.DataFrame,
    gene_symbol: str,
    cancer_types: list[str] | None = None,
    endpoint_label: str = "Overall Survival",
    figsize: tuple[float, float] = (22, 12),
    n_cols: int = 6,
    save_path: str | Path | None = None,
) -> matplotlib.figure.Figure:
    """Faceted Kaplan-Meier survival panels stratified by expression.

    Args:
        survival_df: DataFrame from
            :func:`~tcga_db.plots.data.get_survival_data` with columns
            ``cancer_type``, ``time``, ``event``, ``expression_group``.
        survival_stats: DataFrame from
            :func:`~tcga_db.plots.stats.run_survival_analysis` with columns
            ``cancer_type``, ``p_value``, ``n_high``, ``n_low``, ``stars``.
        gene_symbol: Gene symbol for the figure title.
        cancer_types: Optional subset of cancer types to plot.  If ``None``,
            all cancer types with at least one event in each group are shown.
        endpoint_label: Human-readable survival endpoint name for the y-axis
            (e.g. ``"Overall Survival"``).
        figsize: Figure size ``(width, height)`` in inches.
        n_cols: Number of subplot columns.
        save_path: Optional path to save the figure.

    Returns:
        The completed :class:`matplotlib.figure.Figure`.
    """
    available = sorted(survival_df["cancer_type"].unique())
    if cancer_types is not None:
        plot_types = [ct for ct in cancer_types if ct in available]
    else:
        plot_types = available

    if not plot_types:
        raise ValueError("No cancer types to plot after filtering.")

    n_types = len(plot_types)
    n_rows = (n_types + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes_flat = np.array(axes).flatten()

    stats_lookup = (
        survival_stats.set_index("cancer_type").to_dict("index")
        if survival_stats is not None
        else {}
    )

    kmf = KaplanMeierFitter()

    for idx, cancer_type in enumerate(plot_types):
        ax = axes_flat[idx]
        sub = survival_df[survival_df["cancer_type"] == cancer_type]

        high = sub[sub["expression_group"] == "high"]
        low = sub[sub["expression_group"] == "low"]

        for grp, color, label in [
            (high, _COLOUR_HIGH, "High"),
            (low, _COLOUR_LOW, "Low"),
        ]:
            if grp.empty or grp["event"].sum() == 0:
                continue
            kmf.fit(
                grp["time"],
                event_observed=grp["event"],
                label=label,
            )
            kmf.plot_survival_function(
                ax=ax,
                ci_show=True,
                ci_alpha=_ALPHA_CI,
                color=color,
                linewidth=1.4,
                show_censors=True,
                censor_styles={"ms": 4, "marker": "+"},
            )

        # p-value annotation
        stat = stats_lookup.get(cancer_type, {})
        p_val = stat.get("p_value", np.nan)
        n_high = stat.get("n_high", len(high))
        n_low = stat.get("n_low", len(low))
        stars = stat.get("stars", "")

        if not np.isnan(p_val):
            ax.annotate(
                f"{stars}\np={p_val:.2e}\nn={n_high}/{n_low}",
                xy=(0.97, 0.97),
                xycoords="axes fraction",
                ha="right",
                va="top",
                fontsize=6,
                bbox={
                    "boxstyle": "round,pad=0.2",
                    "fc": "white",
                    "alpha": 0.7,
                },
            )

        ax.set_title(cancer_type, fontsize=7, fontweight="bold", pad=2)
        ax.set_xlabel("Time (days)", fontsize=6)
        ax.set_ylabel("")
        ax.tick_params(labelsize=6)
        ax.legend(fontsize=5, loc="lower left", frameon=False)
        sns.despine(ax=ax)

    # Hide unused axes
    for idx in range(n_types, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    # Shared y-axis label
    for row_idx in range(n_rows):
        axes_flat[row_idx * n_cols].set_ylabel(
            f"{endpoint_label} probability", fontsize=7
        )

    fig.suptitle(
        f"{gene_symbol} — {endpoint_label} by Expression (High vs Low)",
        fontsize=13,
        fontweight="bold",
        y=1.01,
    )
    plt.tight_layout(h_pad=1.5, w_pad=0.5)

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)

    return fig
