"""Layer 1 — DuckDB data access for expression analysis.

All query functions return clean pandas DataFrames ready for analysis.

Expression values in the database are stored as log2(TPM + 0.001) (Toil
recompute format).  The functions here back-transform to raw TPM, so that
calling code always works with real TPM values and can apply whatever
log-transform it prefers (plot layer uses log2(TPM + 1) by convention).

Example:
    >>> db = "data/tcga.duckdb"
    >>> gene_id, symbol = resolve_gene("BRCA1", db)
    >>> df = get_gene_expression(gene_id, db)
    >>> df.head()
"""

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

# Mapping from samples.detailed_category → TCGA cancer type code.
# Covers all 33 TCGA cancer types as stored in the Toil phenotype file.
DISEASE_TO_CODE: dict[str, str] = {
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


def _connect(db_path: str | Path) -> duckdb.DuckDBPyConnection:
    """Open a read-only DuckDB connection.

    Args:
        db_path: Path to the ``.duckdb`` file.

    Returns:
        Open DuckDB connection.
    """
    return duckdb.connect(str(db_path), read_only=True)


def _log2_tpm_to_tpm(values: pd.Series) -> pd.Series:
    """Back-transform Toil log2(TPM + 0.001) → raw TPM, clipped to 0.

    Args:
        values: Series of log2(TPM + 0.001) floats.

    Returns:
        Series of raw TPM values (non-negative).
    """
    return np.maximum(0.0, (2.0**values) - 0.001)


def resolve_gene(
    query: str,
    db_path: str | Path,
) -> tuple[str, str]:
    """Resolve a gene symbol or Ensembl ID to ``(gene_id, gene_symbol)``.

    Looks up the ``genes`` table accepting either a Hugo symbol (e.g.
    ``"BRCA1"``) or an unversioned Ensembl ID (e.g. ``"ENSG00000012048"``).
    Only protein-coding genes are searched.

    Args:
        query: Gene symbol or Ensembl gene ID.
        db_path: Path to the ``.duckdb`` file.

    Returns:
        Tuple of ``(gene_id, gene_symbol)`` where ``gene_id`` is the
        unversioned Ensembl ID used as a column name in the expression tables.

    Raises:
        ValueError: If the gene cannot be found.
    """
    with _connect(db_path) as con:
        row = con.execute(
            """
            SELECT gene_id, gene_symbol
            FROM genes
            WHERE is_protein_coding = TRUE
              AND (gene_symbol = ? OR gene_id = ?)
            LIMIT 1
            """,
            [query, query],
        ).fetchone()

    if row is None:
        raise ValueError(
            f"Gene {query!r} not found in genes table "
            "(searched gene_symbol and gene_id, protein-coding only)."
        )
    return row[0], row[1]


def get_gene_expression(
    gene_id: str,
    db_path: str | Path,
    expression_type: str = "tpm",
    cancer_types: list[str] | None = None,
) -> pd.DataFrame:
    """Fetch per-sample expression for one gene across all 33 TCGA types + GTEx.

    Returns all TCGA tumour samples (all 33 cancer types derived from
    ``detailed_category`` via :data:`DISEASE_TO_CODE`) plus GTEx normal
    samples for cancer types that have a tissue_mapping match.  Cancer types
    without GTEx normals will have ``is_tumour = True`` only — the stats
    layer marks these as ``"nd"``.

    Args:
        gene_id: Unversioned Ensembl ID (e.g. ``"ENSG00000012048"``).
            Use :func:`resolve_gene` to obtain this from a symbol.
        db_path: Path to the ``.duckdb`` file.
        expression_type: ``"tpm"`` or ``"counts"``.  Defaults to ``"tpm"``.
        cancer_types: Optional list of TCGA cancer type codes (e.g.
            ``["BRCA", "LUAD"]``) to restrict results.  If ``None``, all
            33 TCGA types are returned.

    Returns:
        DataFrame with columns ``sample_id``, ``patient_id``,
        ``source_study``, ``cancer_type``, ``primary_site_normalised``,
        ``is_tumour``, ``sample_type_code``, ``tpm``.

    Raises:
        ValueError: If ``expression_type`` is not ``"tpm"`` or ``"counts"``.
        KeyError: If ``gene_id`` is not a column in the expression table.
    """
    if expression_type not in ("tpm", "counts"):
        raise ValueError(
            f"expression_type must be 'tpm' or 'counts', got {expression_type!r}"
        )
    table = (
        "expression_tpm_wide"
        if expression_type == "tpm"
        else "expression_counts_wide"
    )
    col = f'"{gene_id}"'

    # Build the disease-to-code mapping as a DuckDB VALUES clause
    map_rows = ", ".join(
        f"('{disease}', '{code}')" for disease, code in DISEASE_TO_CODE.items()
    )

    # Optional cancer type filter (applied after code lookup)
    cancer_filter_tcga = ""
    cancer_filter_gtex = ""
    params: list[str] = []
    if cancer_types:
        placeholders = ", ".join("?" * len(cancer_types))
        cancer_filter_tcga = f"AND dc.tcga_code IN ({placeholders})"
        cancer_filter_gtex = f"AND tm.tcga_cancer_type IN ({placeholders})"
        params = list(cancer_types) * 2  # used in both halves of UNION

    sql = f"""
        WITH disease_codes(disease, tcga_code) AS (
            VALUES {map_rows}
        )
        -- TCGA tumour + solid-tissue normal samples (all 33 types)
        SELECT
            s.sample_id,
            s.patient_id,
            s.source_study,
            dc.tcga_code           AS cancer_type,
            s.primary_site_normalised,
            s.is_tumour,
            s.sample_type_code,
            e.{col}                AS raw_stored
        FROM {table} e
        JOIN samples s USING (sample_id)
        JOIN disease_codes dc ON s.detailed_category = dc.disease
        WHERE s.source_study = 'TCGA'
          AND e.{col} IS NOT NULL
          {cancer_filter_tcga}

        UNION ALL

        -- GTEx normal samples (only for tissue_mapping-matched types)
        SELECT
            s.sample_id,
            s.patient_id,
            s.source_study,
            tm.tcga_cancer_type    AS cancer_type,
            s.primary_site_normalised,
            s.is_tumour,
            s.sample_type_code,
            e.{col}                AS raw_stored
        FROM {table} e
        JOIN samples s USING (sample_id)
        JOIN tissue_mapping tm
          ON s.primary_site_normalised = tm.primary_site_normalised
        WHERE s.source_study = 'GTEx'
          AND e.{col} IS NOT NULL
          {cancer_filter_gtex}
    """

    with _connect(db_path) as con:
        try:
            df = con.execute(sql, params or None).df()
        except duckdb.CatalogException as exc:
            raise KeyError(
                f"Gene ID {gene_id!r} not found as a column in {table}. "
                "Ensure gene_id is a protein-coding Ensembl ID present in "
                "the expression table."
            ) from exc

    if expression_type == "tpm":
        df["tpm"] = _log2_tpm_to_tpm(df["raw_stored"])
    else:
        df["tpm"] = np.maximum(0.0, (2.0 ** df["raw_stored"]) - 1.0).round()

    df = df.drop(columns=["raw_stored"])
    return df


def get_survival_data(
    gene_id: str,
    db_path: str | Path,
    cancer_types: list[str] | None = None,
    endpoint: str = "os",
) -> pd.DataFrame:
    """Fetch per-patient expression + survival endpoint for one gene.

    Restricted to primary tumour samples only (``sample_type_code = '01'``).
    Patients are split into ``high`` / ``low`` expression groups by the
    per-cancer-type median.

    Args:
        gene_id: Unversioned Ensembl ID.
        db_path: Path to the ``.duckdb`` file.
        cancer_types: Optional list of TCGA cancer type codes to restrict.
        endpoint: Survival endpoint — one of ``"os"``, ``"dss"``,
            ``"dfi"``, ``"pfi"``.  Defaults to ``"os"``.

    Returns:
        DataFrame with columns:

        * ``patient_id``
        * ``cancer_type``
        * ``tpm`` — raw TPM
        * ``log2_tpm`` — log2(TPM + 1) for display / split
        * ``time`` — survival time in days
        * ``event`` — boolean event indicator
        * ``expression_group`` — ``"high"`` or ``"low"`` (per-cancer median)

    Raises:
        ValueError: If ``endpoint`` is not recognised.
    """
    valid_endpoints = {"os", "dss", "dfi", "pfi"}
    if endpoint not in valid_endpoints:
        raise ValueError(
            f"endpoint must be one of {valid_endpoints}, got {endpoint!r}"
        )
    time_col = f"{endpoint}_time"
    event_col = f"{endpoint}_event"

    # cancer_type derived from detailed_category via DISEASE_TO_CODE
    # (patients.cancer_type is unreliable — stored as "UNKNOWN").
    map_rows = ", ".join(
        f"('{disease}', '{code}')" for disease, code in DISEASE_TO_CODE.items()
    )

    cancer_filter = ""
    params: list[str] = []
    if cancer_types:
        placeholders = ", ".join("?" * len(cancer_types))
        cancer_filter = f"AND dc.tcga_code IN ({placeholders})"
        params = list(cancer_types)

    col = f'"{gene_id}"'
    sql = f"""
        WITH disease_codes(disease, tcga_code) AS (
            VALUES {map_rows}
        )
        SELECT
            s.patient_id,
            dc.tcga_code         AS cancer_type,
            e.{col}              AS raw_stored,
            sv.{time_col}        AS time,
            sv.{event_col}       AS event
        FROM expression_tpm_wide e
        JOIN samples s USING (sample_id)
        JOIN disease_codes dc ON s.detailed_category = dc.disease
        JOIN survival sv ON s.patient_id = sv.patient_id
        WHERE s.sample_type_code = '01'
          AND s.source_study = 'TCGA'
          AND e.{col} IS NOT NULL
          AND sv.{time_col} IS NOT NULL
          AND sv.{event_col} IS NOT NULL
          {cancer_filter}
    """

    with _connect(db_path) as con:
        df = con.execute(sql, params or None).df()

    df["tpm"] = _log2_tpm_to_tpm(df["raw_stored"])
    df["log2_tpm"] = np.log2(df["tpm"] + 1.0)
    df = df.drop(columns=["raw_stored"])

    # Per-cancer-type median split
    medians = df.groupby("cancer_type")["log2_tpm"].transform("median")
    df["expression_group"] = np.where(df["log2_tpm"] >= medians, "high", "low")

    return df


def get_tissue_mapping(db_path: str | Path) -> pd.DataFrame:
    """Return the full tissue_mapping table.

    Args:
        db_path: Path to the ``.duckdb`` file.

    Returns:
        DataFrame with columns ``tcga_cancer_type``, ``gtex_tissue``,
        ``primary_site_normalised``, ``description``.
    """
    with _connect(db_path) as con:
        return con.execute(
            "SELECT * FROM tissue_mapping ORDER BY tcga_cancer_type"
        ).df()
