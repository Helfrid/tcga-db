

# TCGA/GTEx Local Database: Project Plan

## Objective

Build a local DuckDB database integrating TCGA tumour and GTEx normal tissue data into a single, queryable resource. The database enables patient-level multi-modal queries linking gene expression, copy number, somatic mutations, mutational signatures, and clinical outcomes. It also serves as an independent validation layer for the [[depmap_graph_design_spec|DepMap co-dependency graph project]] — co-dependency relationships discovered in cell lines can be cross-validated against patient tumour data.

## Data Source: UCSC Toil RNA-seq Recompute

All expression data comes from the c hosted on the [Xena Toil Hub](https://toil.xenahubs.net). This project re-processed all TCGA, GTEx, and TARGET RNA-seq samples through a single uniform pipeline (same aligner, quantification, normalisation), eliminating computational batch effects. This is the standard approach in the literature for tumour-vs-normal comparisons.

### Why GTEx rather than TCGA matched normals?

TCGA's "matched normal" samples are adjacent tissue from cancer patients. They have two problems: scarcity (~700 across all cancer types, many types have almost none) and field cancerisation (adjacent tissue may carry molecular alterations from the tumour microenvironment). GTEx provides ~17,000 samples from genuinely healthy donors across 54 tissue types. The Toil recompute makes TCGA and GTEx directly comparable by processing both through the same pipeline.

GTEx normal tissue complementary to TCGA tumour data is available for 16 primary sites — not all cancer types have a matching GTEx tissue.

## Core Downloads (Xena Toil Hub)

| File | Description | URL |
|------|-------------|-----|
| Gene expression (expected counts) | ~60K genes × ~19K samples, log2(count+1) | `toil-xena-hub.s3.us-east-1.amazonaws.com/download/TcgaTargetGtex_gene_expected_count.gz` |
| Gene expression (TPM) | Same shape, log2(TPM+0.001) | `toil-xena-hub.s3.us-east-1.amazonaws.com/download/TcgaTargetGtex_rsem_gene_tpm.gz` |
| Phenotype / sample annotation | Study (TCGA/GTEx/TARGET), tissue, sample type | `toil-xena-hub.s3.us-east-1.amazonaws.com/download/TcgaTargetGTEX_phenotype.txt.gz` |
| Survival data (TCGA only) | OS, DSS, DFI, PFI (curated Liu et al. 2018 endpoints) | `toil-xena-hub.s3.us-east-1.amazonaws.com/download/TCGA_survival_data` |
| Gene ID mapping | Ensembl → gene symbol (GENCODE v23) | `toil-xena-hub.s3.us-east-1.amazonaws.com/download/probeMap/gencode.v23.annotation.gene.probemap` |

Also mirrored on Zenodo for long-term storage: [zenodo.org/records/10944168](https://zenodo.org/records/10944168).

### Note on gene count

The ~60,000 gene identifiers in the Toil data include ~20K protein-coding genes, ~24K non-coding genes (lncRNAs, miRNAs, etc.), and ~15K pseudogenes — all from GENCODE v23. DepMap only covers the ~20K protein-coding genes (CRISPR targets). The gene ID mapping file includes a `gene_type` column, so filtering to protein-coding at query time is straightforward. Plan is to load all genes and filter at query time rather than at ingestion.
For phase-1 we will narrow the db down and focus only on protein coding genes.

## TCGA Barcode System & Patient-Level Linkage

All TCGA data types are linked at the individual patient level through the barcode system. A barcode like `TCGA-A7-A13E-01A-11R-A13Q-07` encodes: project, patient ID (first 12 characters), sample type (characters 14–15: 01 = primary tumour, 11 = solid tissue normal), portion, analyte, plate/centre. Different assays (RNA-seq, copy number, exome sequencing) on tissue from the same patient share the patient prefix. This means expression, copy number, mutations, mutational signatures, and survival are all joinable per patient.

Coverage: ~11K patients have clinical/survival, ~10K have RNA-seq, ~10K have copy number, ~10K have somatic mutations. Intersection of all four is ~9K+ patients across 33 cancer types.

## Phased Build Plan

### Phase 1: Expression + Survival (Easter 2026)

- Gene-level expression matrix (Toil recompute, only protein coding genes ~20k)
- Sample/phenotype annotations (study, tissue, sample type)
- Clinical/survival endpoints (OS, DSS, DFI, PFI)
- Gene ID mapping table (Ensembl → symbol, with gene_type for filtering)
- Patient-centric schema design with barcode parsing

The database should support queries like: *"What is the expression of BRCA1 in primary tumour vs GTEx normal across all cancer types with matched tissue, and does high expression correlate with survival?"*

**Estimated size:** 4–5 GB (protein-coding filter at query time) to 10–15 GB (all genes). DuckDB columnar compression handles the wide-to-long pivot of the expression matrix efficiently.

**Data gotchas to handle during ingestion:**
- Expression values are log2(x+1) transformed — back-transform with 2^x - 1 if raw counts needed for DESeq2/limma-voom
- Capitalisation inconsistencies between TCGA and GTEx phenotype annotations (e.g. "Adrenal Gland" vs "Adrenal gland") — normalise case at load time
- TCGA barcode parsing to extract patient ID and sample type

### Phase 2: Copy Number + Mutations + Proteomics

- **GISTIC2 thresholded copy number** (gene-level, discrete: -2 to +2) from Xena Pan-Cancer Atlas hub
- **Tumour purity and ploidy estimates** (ABSOLUTE/ASCAT) from clinical annotations
- **Somatic mutations** (MAF format) — SNVs, indels with functional annotation (missense, nonsense, frameshift, etc.)
- **Mutational signatures** — COSMIC signature exposures per patient (signature 3 = HR deficiency, signature 1 = ageing, signature 4 = smoking, etc.)
- **MSI status** per sample
- **Proteomics** — TCGA RPPA data (~200 proteins, limited coverage) and/or CPTAC mass-spec proteomics (~10K proteins, subset of cancer types, separate project with partial TCGA patient overlap)

**Estimated additional size:** 2–3 GB. Enables compound queries like: *"Show me genes that are amplified AND overexpressed in breast tumours carrying mutational signature 3."*

### Phase 3: Isoform-Level Expression

- Transcript-level expression from Toil recompute (~200K transcripts × ~19K samples)
- Transcript → gene mapping (GENCODE v23)
- Enables differential transcript usage and splice-switching analyses

**Estimated additional size:** 10–15 GB (protein-coding transcripts) to 25–40 GB (all transcripts).

## Relationship to DepMap Graph Project

The TCGA/GTEx database is a **separate DuckDB file** from the DepMap database. DuckDB supports cross-database queries via `ATTACH`, so joins between the two are seamless without merging schemas.

The key analytical connection: co-dependency relationships discovered in the DepMap graph can be validated against patient tumour data. For example, if the graph reveals a novel co-dependency cluster, we can check whether those genes show correlated expression changes, co-amplification, or coordinated survival associations in patient tumours. This multi-modal cross-validation strengthens biological claims in the graph project publication.

## Publication Strategy

The database alone is not a publishable unit — the data is public and the technology is standard. However:

- **As infrastructure for the graph paper:** The TCGA/GTEx database provides a patient-level validation layer for DepMap graph discoveries, strengthening the main publication.
- **As a tool/application note:** If a proper CLI or Python API is built (enabling cross-resource queries in a single call), a short application note in *Bioinformatics* or *JOSS* is feasible alongside the graph paper.
- **As a resource paper:** If combined with systematic pan-cancer analysis (e.g. identifying genes that are both selectively essential in DepMap AND differentially expressed with survival impact), *NAR Database Issue* or *Cancer Research* would be appropriate venues.

## Tools & References

- [UCSC Xena Browser](https://xenabrowser.net/) — interactive exploration and data download
- [xenaPython](https://github.com/ucscXena/xenaPython) — Python API for Xena Hubs
- [UCSCXenaTools](https://github.com/ropensci/UCSCXenaTools) — R package (Wang et al., JOSS 2019)
- [awesome-TCGA](https://github.com/IARCbioinfo/awesome-TCGA) — curated list of TCGA resources
- Vivian et al. (2017) *Nature Biotechnology* 35:314–316 — Toil recompute publication
- Liu et al. (2018) *Cell* — curated TCGA survival endpoints
- Goldman et al. (2020) *Nature Biotechnology* — UCSC Xena platform paper

## Related

- [[@TCGA]] — project MOC
- [[@DepMap]] — DepMap graph project MOC
- [[depmap_graph_design_spec]] — graph architecture (validation layer connection)
- [[depmap_use_cases]] — use cases that could leverage TCGA validation
