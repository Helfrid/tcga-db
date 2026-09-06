"""Interactive TCGA tumour-vs-normal explorer.

Run as an app:

    uv run marimo run notebooks/tcga_explorer.py

Or edit reactively:

    uv run marimo edit notebooks/tcga_explorer.py
"""

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        # TCGA expression & survival explorer

        Tumour (TCGA) vs healthy normal (GTEx) expression across cancer types,
        with Kaplan-Meier survival stratified by expression.

        Backed by the local `tcga.duckdb` — 18,397 samples, 33 cancer types,
        uniformly processed through the UCSC Toil recompute so that tumour and
        normal are directly comparable.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    gene = mo.ui.text(value="MASTL", label="Gene symbol or Ensembl ID")
    endpoint = mo.ui.dropdown(
        options={
            "Overall survival": "os",
            "Disease-specific survival": "dss",
            "Disease-free interval": "dfi",
            "Progression-free interval": "pfi",
        },
        value="Overall survival",
        label="Survival endpoint",
    )
    run = mo.ui.run_button(label="Analyse")
    mo.hstack([gene, endpoint, run], justify="start", gap=2)
    return endpoint, gene, run


@app.cell
def _(DB_PATH, analyse_gene, endpoint, gene, mo, run):
    mo.stop(
        not run.value,
        mo.md("*Choose a gene and press **Analyse**.*"),
    )
    with mo.status.spinner(title=f"Analysing {gene.value}..."):
        results = analyse_gene(
            gene.value.strip(),
            db_path=DB_PATH,
            survival_endpoint=endpoint.value,
        )
    return (results,)


@app.cell(hide_code=True)
def _(mo, results):
    _de = results["de_results"]
    _sig = _de[_de["status"] == "significant"]
    mo.hstack(
        [
            mo.stat(label="Gene", value=results["gene_symbol"]),
            mo.stat(label="Types tested", value=str(len(_de))),
            mo.stat(label="Significant", value=str(len(_sig))),
            mo.stat(
                label="Up in tumour",
                value=str(int((_sig["log2fc"] > 0).sum())),
            ),
            mo.stat(
                label="Down in tumour",
                value=str(int((_sig["log2fc"] < 0).sum())),
            ),
        ],
        justify="start",
        gap=2,
    )
    return


@app.cell(hide_code=True)
def _(mo, results):
    mo.md(
        f"## Expression — {results['gene_symbol']} tumour vs normal\n\n"
        "Grey = GTEx normal, red = TCGA tumour. Stars are Wilcoxon rank-sum "
        "significance; `nd` marks cancer types with no matched GTEx tissue."
    )
    return


@app.cell(hide_code=True)
def _(results):
    results["expression_fig"]
    return


@app.cell(hide_code=True)
def _(mo, results):
    mo.md(
        f"## Survival — {results['gene_symbol']} high vs low expression\n\n"
        "Median split within each cancer type, log-rank test."
    )
    return


@app.cell(hide_code=True)
def _(results):
    results["survival_fig"]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("## Differential expression statistics")
    return


@app.cell(hide_code=True)
def _(mo, results):
    mo.ui.table(
        results["de_results"].sort_values("padj"),
        selection=None,
        page_size=10,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("## Survival statistics")
    return


@app.cell(hide_code=True)
def _(mo, results):
    mo.ui.table(
        results["survival_results"].sort_values("p_value"),
        selection=None,
        page_size=10,
    )
    return


@app.cell
def _():
    import os
    from pathlib import Path

    import marimo as mo

    from tcga_db.plots import analyse_gene

    # marimo runs from the project root; allow an override elsewhere.
    DB_PATH = os.environ.get(
        "TCGA_DB_PATH", str(Path.cwd() / "data" / "tcga.duckdb")
    )
    if not Path(DB_PATH).exists():
        raise FileNotFoundError(
            f"tcga.duckdb not found at {DB_PATH}. "
            "Run from the project root or set TCGA_DB_PATH."
        )
    return DB_PATH, analyse_gene, mo


if __name__ == "__main__":
    app.run()
