"""Layer 3 — pan-cancer expression violin + jitter plot.

Generates the reference-style figure: one subplot per TCGA cancer type,
each showing Normal (N, grey) vs Tumour (T, pink/red) side-by-side with
violin + jitter, quartile dashed lines, and significance annotation.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

if TYPE_CHECKING:
    import matplotlib.axes
    import matplotlib.figure


# ── Colour scheme ────────────────────────────────────────────────────────────
_COLOUR_NORMAL = "#9E9E9E"  # medium grey
_COLOUR_TUMOUR = "#E57373"  # muted red-pink
_ALPHA_JITTER = 0.45
_JITTER_SIZE = 1.8
_VIOLIN_ALPHA = 0.75


def _add_quartile_lines(
    ax: matplotlib.axes.Axes,
    values: pd.Series,
    x_pos: float,
    width: float = 0.25,
    color: str = "black",
) -> None:
    """Draw dashed lines at 25th, 50th (median), 75th percentiles.

    Args:
        ax: Axes to draw on.
        values: Data values.
        x_pos: X centre position of the violin.
        width: Half-width of each dashed line.
        color: Line colour.
    """
    if values.empty:
        return
    for q in [0.25, 0.50, 0.75]:
        y = float(np.quantile(values.dropna(), q))
        lw = 1.5 if q == 0.50 else 0.8
        ax.plot(
            [x_pos - width, x_pos + width],
            [y, y],
            color=color,
            linewidth=lw,
            linestyle="--",
            zorder=4,
        )


def _add_median_dot(
    ax: matplotlib.axes.Axes, values: pd.Series, x_pos: float
) -> None:
    """Draw a white circle with black border at the median.

    Args:
        ax: Axes to draw on.
        values: Data values.
        x_pos: X centre position.
    """
    if values.empty:
        return
    median = float(np.median(values.dropna()))
    ax.scatter(
        [x_pos],
        [median],
        color="white",
        edgecolors="black",
        linewidths=0.8,
        s=18,
        zorder=5,
    )


def _significance_color(status: str) -> str:
    """Return annotation colour for a significance status.

    Args:
        status: ``"significant"``, ``"ns"``, or ``"nd"``.

    Returns:
        Hex colour string.
    """
    return "#555555" if status == "nd" else "black"


def plot_pan_cancer_expression(
    expression_df: pd.DataFrame,
    stats_df: pd.DataFrame,
    gene_symbol: str,
    figsize: tuple[float, float] = (22, 12),
    n_cols: int = 11,
    save_path: str | Path | None = None,
) -> matplotlib.figure.Figure:
    """Generate the reference-style pan-cancer expression figure.

    Creates a faceted grid of violin + jitter plots, one subplot per TCGA
    cancer type, showing Normal (N) vs Tumour (T) expression on a shared
    log2(TPM + 1) y-axis.

    Args:
        expression_df: DataFrame from
            :func:`~tcga_db.plots.data.get_gene_expression` with columns
            ``cancer_type``, ``is_tumour``, ``tpm``.
        stats_df: DataFrame from
            :func:`~tcga_db.plots.stats.run_differential_expression` with
            columns ``cancer_type``, ``stars``, ``status``.
        gene_symbol: Gene symbol used for the figure title and y-axis label.
        figsize: Overall figure size ``(width, height)`` in inches.
        n_cols: Number of subplot columns.  Rows are computed automatically.
        save_path: If given, the figure is saved to this path.  The format
            is inferred from the extension (e.g. ``.pdf``, ``.png``,
            ``.svg``).

    Returns:
        The completed :class:`matplotlib.figure.Figure`.
    """
    cancer_types = sorted(expression_df["cancer_type"].unique())
    n_types = len(cancer_types)
    n_rows = (n_types + n_cols - 1) // n_cols

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=figsize,
        sharey=True,
    )
    axes_flat = axes.flatten()

    # Precompute log2 transform once
    df = expression_df.copy()
    df["log2_tpm"] = np.log2(df["tpm"] + 1.0)

    stats_lookup = (
        stats_df.set_index("cancer_type")[["stars", "status"]].to_dict("index")
        if stats_df is not None
        else {}
    )

    for idx, cancer_type in enumerate(cancer_types):
        ax = axes_flat[idx]
        sub = df[df["cancer_type"] == cancer_type]

        normal = sub.loc[~sub["is_tumour"], "log2_tpm"]
        tumour = sub.loc[sub["is_tumour"], "log2_tpm"]

        # Build a tidy frame for seaborn violin
        plot_data = pd.concat(
            [
                pd.DataFrame({"value": normal, "group": "N"}),
                pd.DataFrame({"value": tumour, "group": "T"}),
            ]
        )

        # Violin (hue= required in seaborn >=0.13 to use palette)
        sns.violinplot(
            data=plot_data,
            x="group",
            y="value",
            hue="group",
            order=["N", "T"],
            hue_order=["N", "T"],
            palette={"N": _COLOUR_NORMAL, "T": _COLOUR_TUMOUR},
            inner=None,
            cut=0,
            linewidth=0.6,
            alpha=_VIOLIN_ALPHA,
            ax=ax,
            density_norm="width",
            legend=False,
        )

        # Jitter
        rng = np.random.default_rng(seed=42)
        for x_pos, values in [(0, normal), (1, tumour)]:
            color = _COLOUR_NORMAL if x_pos == 0 else _COLOUR_TUMOUR
            if values.empty:
                continue
            jitter = rng.uniform(-0.12, 0.12, size=len(values))
            ax.scatter(
                x_pos + jitter,
                values,
                color=color,
                alpha=_ALPHA_JITTER,
                s=_JITTER_SIZE,
                linewidths=0,
                zorder=3,
            )

        # Quartile lines + median dot
        for x_pos, values in [(0, normal), (1, tumour)]:
            color = _COLOUR_NORMAL if x_pos == 0 else _COLOUR_TUMOUR
            _add_quartile_lines(ax, values, x_pos, color=color)
            _add_median_dot(ax, values, x_pos)

        # Significance annotation
        sig = stats_lookup.get(cancer_type, {"stars": "nd", "status": "nd"})
        ax.set_title(
            cancer_type,
            fontsize=7,
            fontweight="bold",
            pad=2,
        )
        ax.annotate(
            sig["stars"],
            xy=(0.5, 1.0),
            xycoords="axes fraction",
            xytext=(0, 2),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=7,
            color=_significance_color(sig["status"]),
        )

        ax.set_xlabel("")
        ax.tick_params(axis="both", labelsize=6)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["N", "T"], fontsize=7)
        sns.despine(ax=ax, left=False, bottom=False)

    # Hide unused subplots
    for idx in range(n_types, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    # Shared y-axis label on the leftmost column
    for row_idx in range(n_rows):
        axes_flat[row_idx * n_cols].set_ylabel("log₂(TPM + 1)", fontsize=8)

    fig.suptitle(
        f"{gene_symbol} Expression — TCGA Tumour vs GTEx Normal",
        fontsize=13,
        fontweight="bold",
        y=1.01,
    )
    fig.text(
        0.5,
        -0.01,
        "Tissue (N – Normal, T – Tumour)",
        ha="center",
        fontsize=9,
    )

    plt.tight_layout(h_pad=1.5, w_pad=0.5)

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)

    return fig
