"""Layer 2 — statistical analysis for expression and survival.

Primary method: Wilcoxon rank-sum on log2(TPM + 1).
This avoids the source_study / condition confounding that arises with
DESeq2 when all tumours are TCGA and all normals are GTEx.

Survival analysis uses log-rank test via lifelines.
"""

import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from scipy.stats import mannwhitneyu

# Minimum samples per group to attempt a statistical test
_MIN_SAMPLES = 3

# Significance star thresholds (padj / p-value)
_STAR_THRESHOLDS = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 5e-2]
_STAR_LABELS = ["******", "*****", "****", "***", "**", "*"]


def pvalue_to_stars(p: float) -> str:
    """Convert a p-value to a significance star string.

    Args:
        p: p-value or adjusted p-value.

    Returns:
        One of ``"******"`` through ``"*"``, ``"ns"`` (not significant),
        following the convention ``* p<0.05`` … ``****** p<1e-6``.
    """
    for threshold, label in zip(_STAR_THRESHOLDS, _STAR_LABELS, strict=True):
        if p < threshold:
            return label
    return "ns"


def run_differential_expression(
    expression_df: pd.DataFrame,
    min_samples: int = _MIN_SAMPLES,
) -> pd.DataFrame:
    """Run per-cancer-type tumour-vs-normal differential expression.

    Uses a two-sided Wilcoxon rank-sum (Mann-Whitney U) test on
    log2(TPM + 1) values.  Multiple-testing correction is applied across
    cancer types using Benjamini-Hochberg FDR.

    Args:
        expression_df: DataFrame from :func:`~tcga_db.plots.data.get_gene_expression`
            with at least columns ``cancer_type``, ``is_tumour``, ``tpm``.
        min_samples: Minimum number of samples in each group (tumour AND
            normal) to attempt a test.  Groups below this threshold are
            assigned status ``"nd"`` (no data).

    Returns:
        DataFrame with one row per cancer type and columns:

        * ``cancer_type``
        * ``log2fc`` — median log2(TPM+1) tumour − normal
        * ``p_value`` — raw Wilcoxon p-value
        * ``padj`` — Benjamini-Hochberg adjusted p-value
        * ``n_tumour``
        * ``n_normal``
        * ``median_tumour`` — median log2(TPM+1) in tumour
        * ``median_normal`` — median log2(TPM+1) in normal
        * ``stars`` — significance annotation string
        * ``status`` — ``"significant"``, ``"ns"``, or ``"nd"``
    """
    df = expression_df.copy()
    df["log2_tpm"] = np.log2(df["tpm"] + 1.0)

    results: list[dict] = []
    for cancer_type, group in df.groupby("cancer_type"):
        tumour = group.loc[group["is_tumour"], "log2_tpm"].dropna()
        normal = group.loc[~group["is_tumour"], "log2_tpm"].dropna()

        n_t, n_n = len(tumour), len(normal)
        row: dict = {
            "cancer_type": cancer_type,
            "n_tumour": n_t,
            "n_normal": n_n,
            "median_tumour": float(tumour.median()) if n_t > 0 else np.nan,
            "median_normal": float(normal.median()) if n_n > 0 else np.nan,
            "log2fc": np.nan,
            "p_value": np.nan,
            "padj": np.nan,
            "stars": "nd",
            "status": "nd",
        }

        if n_t < min_samples or n_n < min_samples:
            results.append(row)
            continue

        row["log2fc"] = row["median_tumour"] - row["median_normal"]

        stat, p = mannwhitneyu(tumour, normal, alternative="two-sided")
        row["p_value"] = float(p)
        results.append(row)

    result_df = pd.DataFrame(results)

    # Benjamini-Hochberg FDR correction on testable rows
    testable = result_df["p_value"].notna()
    if testable.any():
        p_values = result_df.loc[testable, "p_value"].values
        result_df.loc[testable, "padj"] = _bh_correction(p_values)

    # Assign stars and status
    def _annotate(row: pd.Series) -> pd.Series:
        if pd.isna(row["padj"]):
            row["stars"] = "nd"
            row["status"] = "nd"
        elif row["padj"] < 0.05:
            row["stars"] = pvalue_to_stars(row["padj"])
            row["status"] = "significant"
        else:
            row["stars"] = "ns"
            row["status"] = "ns"
        return row

    result_df = result_df.apply(_annotate, axis=1)
    return result_df.sort_values("cancer_type").reset_index(drop=True)


def _bh_correction(p_values: np.ndarray) -> np.ndarray:
    """Apply Benjamini-Hochberg FDR correction.

    Args:
        p_values: 1-D array of raw p-values.

    Returns:
        Array of adjusted p-values, clipped to [0, 1].
    """
    n = len(p_values)
    order = np.argsort(p_values)
    ranks = np.empty(n, dtype=int)
    ranks[order] = np.arange(1, n + 1)
    adjusted = p_values * n / ranks
    # Enforce monotonicity from the right
    adjusted = np.minimum.accumulate(adjusted[order][::-1])[::-1]
    result = np.empty(n)
    result[order] = adjusted
    return np.clip(result, 0.0, 1.0)


def run_survival_analysis(
    survival_df: pd.DataFrame,
    min_samples: int = _MIN_SAMPLES,
) -> pd.DataFrame:
    """Per-cancer-type survival analysis stratified by expression group.

    Uses a log-rank test comparing high vs low expression (median split,
    already computed in :func:`~tcga_db.plots.data.get_survival_data`).

    Args:
        survival_df: DataFrame from
            :func:`~tcga_db.plots.data.get_survival_data` with columns
            ``cancer_type``, ``time``, ``event``, ``expression_group``,
            ``log2_tpm``.
        min_samples: Minimum samples per group to attempt a test.

    Returns:
        DataFrame with one row per cancer type and columns:

        * ``cancer_type``
        * ``p_value`` — log-rank p-value
        * ``n_high``, ``n_low``
        * ``median_survival_high``, ``median_survival_low`` — days
        * ``median_split`` — log2(TPM+1) split value used
        * ``status`` — ``"significant"``, ``"ns"``, or ``"nd"``
        * ``stars``
    """
    results: list[dict] = []

    for cancer_type, group in survival_df.groupby("cancer_type"):
        high = group[group["expression_group"] == "high"]
        low = group[group["expression_group"] == "low"]
        n_high, n_low = len(high), len(low)

        row: dict = {
            "cancer_type": cancer_type,
            "n_high": n_high,
            "n_low": n_low,
            "median_split": float(group["log2_tpm"].median()),
            "p_value": np.nan,
            "median_survival_high": np.nan,
            "median_survival_low": np.nan,
            "stars": "nd",
            "status": "nd",
        }

        if n_high < min_samples or n_low < min_samples:
            results.append(row)
            continue

        # Log-rank test
        result = logrank_test(
            high["time"],
            low["time"],
            event_observed_A=high["event"],
            event_observed_B=low["event"],
        )
        row["p_value"] = float(result.p_value)

        # Median survival per group
        for label, grp in [("high", high), ("low", low)]:
            kmf = KaplanMeierFitter()
            kmf.fit(grp["time"], event_observed=grp["event"])
            row[f"median_survival_{label}"] = float(kmf.median_survival_time_)

        row["stars"] = pvalue_to_stars(row["p_value"])
        row["status"] = "significant" if row["p_value"] < 0.05 else "ns"
        results.append(row)

    result_df = pd.DataFrame(results)
    if result_df.empty:
        return result_df
    return result_df.sort_values("cancer_type").reset_index(drop=True)
