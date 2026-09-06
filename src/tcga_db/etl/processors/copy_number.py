"""Copy number processor — GISTIC2 thresholded gene-level calls.

Loads the Xena Pan-Cancer Atlas GISTIC2 file (gene × TCGA-sample matrix)
into a wide Parquet file + DuckDB VIEW, following the same pattern as the
expression matrix processors.

Source: broad.mit.edu_PANCAN_Genome_Wide_SNP_6_whitelisted.gene.xena.gz
Target: copy_number_wide (wide Parquet + DuckDB VIEW)

Values are GISTIC2 thresholded discrete calls:
  -2 = homozygous deletion
  -1 = heterozygous deletion
   0 = copy neutral
  +1 = low-level gain
  +2 = high-level amplification

Only TCGA primary tumour samples are included (no GTEx).
Gene lookup uses Hugo symbols (the identifier in the GISTIC2 file).
"""

import gzip
import shutil
import time
from pathlib import Path

import pandas as pd

from ...config import get_logger
from ..base import BaseProcessor, ProcessingResult

logger = get_logger(__name__)

_SAMPLES_PER_CHUNK = 2000
_PIVOT_BATCH = 500


class CopyNumberWideProcessor(BaseProcessor):
    """Load GISTIC2 thresholded copy number into a wide Parquet + VIEW.

    Filters to protein-coding genes (joined by gene symbol) and to
    TCGA tumour samples only.

    Args:
        batch_size: INSERT batch size (unused for Parquet path but kept
            for interface consistency).
    """

    def __init__(self, batch_size: int | None = None) -> None:
        super().__init__(dataset_name="copy_number", batch_size=batch_size)

    def get_table_name(self) -> str:
        """Return target table / VIEW name."""
        return "copy_number_wide"

    def validate_data(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, list[str]]:
        return df, []

    def transform_data(self, df: pd.DataFrame) -> pd.DataFrame:
        return df

    # ------------------------------------------------------------------
    # Overrides for Parquet-backed wide table
    # ------------------------------------------------------------------

    def _get_parquet_path(self) -> Path:
        """Path to the wide Parquet output file."""
        return Path("data/copy_number_wide.parquet")

    def _get_protein_coding_gene_symbols(self) -> dict[str, str]:
        """Return {gene_symbol: gene_id} for protein-coding genes.

        Returns:
            Mapping from Hugo gene symbol to unversioned Ensembl ID.
        """
        result = self.db_manager.execute(
            "SELECT gene_symbol, gene_id FROM genes "
            "WHERE is_protein_coding = TRUE"
        )
        return {row[0]: row[1] for row in result.fetchall()}

    def _get_valid_tumour_sample_ids(self) -> set[str]:
        """Return sample IDs for TCGA primary tumour samples.

        Returns:
            Set of sample_id strings matching TCGA tumour samples.
        """
        result = self.db_manager.execute(
            "SELECT sample_id FROM samples "
            "WHERE source_study = 'TCGA' AND is_tumour = TRUE"
        )
        return {row[0] for row in result.fetchall()}

    @staticmethod
    def _read_tsv_header(file_path: Path) -> list[str]:
        """Read the first line of a (possibly gzipped) TSV to get columns.

        Args:
            file_path: Path to the TSV or TSV.gz file.

        Returns:
            List of column header strings.
        """
        opener = gzip.open if str(file_path).endswith(".gz") else open
        with opener(file_path, "rt") as fh:  # type: ignore[call-overload]
            header = fh.readline().strip()
        return header.split("\t")

    def get_existing_record_count(self) -> int:
        """Return sample count from the Parquet file if it exists.

        Returns:
            Number of samples already loaded, or 0 if no Parquet file.
        """
        parquet = self._get_parquet_path()
        if parquet.exists():
            result = self.db_manager.execute(
                f"SELECT COUNT(*) FROM read_parquet('{parquet}')"
            )
            return result.fetchone()[0]  # type: ignore[index]
        return 0

    # ------------------------------------------------------------------
    # Main processing
    # ------------------------------------------------------------------

    def process_file(
        self, file_path: Path, force_reload: bool = False
    ) -> ProcessingResult:
        """Load GISTIC2 copy number into wide Parquet + DuckDB VIEW.

        Steps mirror the expression matrix pipeline:
        1. Read column headers from the source file.
        2. Filter columns to TCGA tumour samples present in the DB.
        3. Unpivot in chunks, join to protein-coding genes by symbol.
        4. Export long staging table to Parquet.
        5. Pivot in pandas batches → write wide Parquet.
        6. Replace schema-init placeholder TABLE with a VIEW.

        Args:
            file_path: Path to the GISTIC2 TSV.gz file.
            force_reload: Delete and rebuild the Parquet if ``True``.

        Returns:
            :class:`~tcga_db.etl.base.ProcessingResult` with counts/timing.
        """
        start = time.time()
        table_name = self.get_table_name()
        parquet_out = self._get_parquet_path()
        warnings: list[str] = []

        existing = self.get_existing_record_count()
        if existing > 0 and not force_reload:
            return ProcessingResult(
                processor_name=self.__class__.__name__,
                dataset_name=self.dataset_name,
                source_file=file_path,
                records_processed=0,
                records_inserted=0,
                records_skipped=existing,
                processing_time_seconds=time.time() - start,
                status="skipped",
            )

        if force_reload and parquet_out.exists():
            parquet_out.unlink()

        # Prerequisite checks
        pc_genes = self._get_protein_coding_gene_symbols()
        if not pc_genes:
            raise ValueError(
                "No protein-coding genes found. Run gene_probemap load first."
            )
        self.logger.info("Found %d protein-coding genes", len(pc_genes))

        valid_samples = self._get_valid_tumour_sample_ids()
        if not valid_samples:
            raise ValueError(
                "No TCGA tumour samples found. Run phenotype load first."
            )
        self.logger.info(
            "Found %d valid TCGA tumour samples", len(valid_samples)
        )

        # Tune DuckDB for large matrix operations
        self.db_manager.execute("SET memory_limit = '4GB'")
        self.db_manager.execute("SET preserve_insertion_order = false")
        self.db_manager.execute("SET threads = 2")

        # Discover column layout from header
        all_cols = self._read_tsv_header(file_path)
        gene_col = all_cols[0]
        file_sample_cols = all_cols[1:]

        sample_cols = [c for c in file_sample_cols if c in valid_samples]
        self.logger.info(
            "Matched %d / %d sample columns",
            len(sample_cols),
            len(file_sample_cols),
        )
        warnings.append(
            f"Matched {len(sample_cols)} of {len(file_sample_cols)} "
            f"sample columns"
        )

        if not sample_cols:
            raise ValueError(
                "No sample columns matched TCGA tumour samples in the DB. "
                "Ensure the phenotype data is loaded before copy number."
            )

        # Protein-coding gene symbol lookup
        self.db_manager.execute(
            """
            CREATE OR REPLACE TEMP TABLE pc_gene_syms AS
            SELECT gene_symbol, gene_id
            FROM genes WHERE is_protein_coding = TRUE
            """
        )

        # Staging long table
        self.db_manager.execute(
            """
            CREATE OR REPLACE TEMP TABLE cn_long (
                gene_id VARCHAR,
                sample_id VARCHAR,
                value DOUBLE
            )
            """
        )

        # Chunked unpivot
        chunks = [
            sample_cols[i : i + _SAMPLES_PER_CHUNK]
            for i in range(0, len(sample_cols), _SAMPLES_PER_CHUNK)
        ]
        self.logger.info(
            "Processing %d samples in %d chunks of %d",
            len(sample_cols),
            len(chunks),
            _SAMPLES_PER_CHUNK,
        )

        total_rows = 0
        for i, chunk_cols in enumerate(chunks):
            self.logger.info(
                "Chunk %d/%d: %d samples...",
                i + 1,
                len(chunks),
                len(chunk_cols),
            )
            quoted_chunk = ", ".join(f'"{c}"' for c in chunk_cols)
            select_cols = f'"{gene_col}", {quoted_chunk}'

            self.db_manager.execute(
                f"""
                INSERT INTO cn_long
                SELECT
                    g.gene_id,
                    u.sample_id,
                    TRY_CAST(u.value AS DOUBLE) AS value
                FROM (
                    UNPIVOT (
                        SELECT {select_cols}
                        FROM read_csv(
                            '{file_path}',
                            delim='\\t',
                            header=true,
                            all_varchar=true
                        )
                    )
                    ON {quoted_chunk}
                    INTO NAME sample_id VALUE value
                ) u
                INNER JOIN pc_gene_syms g
                    ON u."{gene_col}" = g.gene_symbol
                """
            )

            total_rows = self.db_manager.execute(
                "SELECT COUNT(*) FROM cn_long"
            ).fetchone()[0]  # type: ignore[index]
            self.logger.info("  Cumulative long rows: %d", total_rows)

        self.logger.info("Total long format rows: %d", total_rows)

        actual_genes: list[str] = [
            row[0]
            for row in self.db_manager.execute(
                "SELECT DISTINCT gene_id FROM cn_long ORDER BY gene_id"
            ).fetchall()
        ]
        warnings.append(f"Matched {len(actual_genes)} protein-coding genes")

        # Export long → Parquet
        tmp_dir = Path(f"data/tmp/{table_name}")
        tmp_dir.mkdir(parents=True, exist_ok=True)
        long_parquet = tmp_dir / "long.parquet"

        self.logger.info("Writing long table to Parquet...")
        self.db_manager.execute(
            f"COPY cn_long TO '{long_parquet}' (FORMAT PARQUET)"
        )
        self.db_manager.execute("DROP TABLE IF EXISTS cn_long")
        self.db_manager.execute("DROP TABLE IF EXISTS pc_gene_syms")

        # Pivot in pandas batches → wide Parquet
        sample_list: list[str] = [
            row[0]
            for row in self.db_manager.execute(
                f"SELECT DISTINCT sample_id "
                f"FROM read_parquet('{long_parquet}') ORDER BY sample_id"
            ).fetchall()
        ]
        self.logger.info(
            "Pivoting %d samples to wide Parquet...", len(sample_list)
        )

        wide_parts: list[pd.DataFrame] = []
        for i in range(0, len(sample_list), _PIVOT_BATCH):
            batch_samples = sample_list[i : i + _PIVOT_BATCH]
            placeholders = ", ".join(f"'{s}'" for s in batch_samples)

            batch_df = self.db_manager.execute(
                f"SELECT * FROM read_parquet('{long_parquet}') "
                f"WHERE sample_id IN ({placeholders})"
            ).df()

            wide = batch_df.pivot(
                index="sample_id", columns="gene_id", values="value"
            ).reset_index()
            wide_parts.append(wide)

            if (i // _PIVOT_BATCH) % 5 == 0:
                self.logger.info(
                    "  Pivoted %d / %d samples",
                    min(i + _PIVOT_BATCH, len(sample_list)),
                    len(sample_list),
                )

        self.logger.info("Writing final wide Parquet to %s", parquet_out)
        full_wide = pd.concat(wide_parts, ignore_index=True)
        full_wide.to_parquet(parquet_out, index=False)
        inserted = len(full_wide)
        del full_wide, wide_parts

        # Replace schema placeholder TABLE with VIEW
        self.db_manager.execute(f"DROP TABLE IF EXISTS {table_name}")
        self.db_manager.execute(f"DROP VIEW IF EXISTS {table_name}")
        abs_parquet = parquet_out.resolve()
        self.db_manager.execute(
            f"""
            CREATE VIEW {table_name} AS
            SELECT * FROM read_parquet('{abs_parquet}')
            """
        )

        shutil.rmtree(tmp_dir, ignore_errors=True)

        elapsed = time.time() - start
        self.logger.info(
            "Loaded %d samples × %d genes into %s in %.1fs",
            inserted,
            len(actual_genes),
            table_name,
            elapsed,
        )
        self._record_import(file_path, inserted, "success")

        return ProcessingResult(
            processor_name=self.__class__.__name__,
            dataset_name=self.dataset_name,
            source_file=file_path,
            records_processed=total_rows,
            records_inserted=inserted,
            records_skipped=0,
            processing_time_seconds=elapsed,
            status="success",
            warnings=warnings,
        )
