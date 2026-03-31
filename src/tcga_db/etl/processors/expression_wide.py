"""Expression matrix processors — chunked loading to Parquet.

Handles the large Toil expression matrices (~60K genes × ~19K samples)
by processing in sample-column chunks and storing as wide Parquet files.
DuckDB views provide SQL access without the ~20K-column table overhead
that causes OOM on commit.

Strategy:
1. Read TSV in chunks of sample columns
2. Filter to protein-coding genes via JOIN, unpivot to long format
3. Export long table to Parquet
4. Pivot in pandas batches → write wide Parquet
5. Create DuckDB VIEW pointing to the Parquet file
"""

import gzip
import shutil
import time
from pathlib import Path

import pandas as pd

from ...config import get_logger
from ..base import BaseProcessor, ProcessingResult

logger = get_logger(__name__)

SAMPLES_PER_CHUNK = 3000


class ExpressionWideProcessor(BaseProcessor):
    """Base class for loading wide-format expression matrices.

    Stores the final wide matrix as a Parquet file in ``data/``
    and creates a DuckDB VIEW for SQL access. This avoids the
    DuckDB commit OOM that occurs with ~20K-column tables on
    machines with <=16GB RAM.
    """

    def __init__(
        self, dataset_name: str, batch_size: int | None = None
    ) -> None:
        super().__init__(dataset_name=dataset_name, batch_size=batch_size)

    def get_table_name(self) -> str:
        raise NotImplementedError

    def validate_data(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, list[str]]:
        return df, []

    def transform_data(self, df: pd.DataFrame) -> pd.DataFrame:
        return df

    def _get_parquet_path(self) -> Path:
        """Path to the wide Parquet file for this expression type."""
        return Path(f"data/{self.get_table_name()}.parquet")

    def _get_protein_coding_genes(self) -> list[str]:
        result = self.db_manager.execute(
            "SELECT gene_id FROM genes WHERE is_protein_coding = TRUE"
        )
        return [row[0] for row in result.fetchall()]

    def _get_valid_sample_ids(self) -> set[str]:
        result = self.db_manager.execute("SELECT sample_id FROM samples")
        return {row[0] for row in result.fetchall()}

    def _read_tsv_header(self, file_path: Path) -> list[str]:
        opener = gzip.open if str(file_path).endswith(".gz") else open
        with opener(file_path, "rt") as fh:
            header = fh.readline().strip()
        return header.split("\t")

    def get_existing_record_count(self) -> int:
        """Check if Parquet file exists (overrides base table check)."""
        parquet = self._get_parquet_path()
        if parquet.exists():
            result = self.db_manager.execute(
                f"SELECT COUNT(*) FROM read_parquet('{parquet}')"
            )
            return result.fetchone()[0]
        return 0

    def process_file(
        self, file_path: Path, force_reload: bool = False
    ) -> ProcessingResult:
        """Load expression matrix to wide Parquet + DuckDB VIEW."""
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

        # Get protein-coding genes and valid samples
        pc_genes = self._get_protein_coding_genes()
        if not pc_genes:
            raise ValueError(
                "No protein-coding genes in genes table. Load genes first."
            )
        self.logger.info("Found %d protein-coding genes", len(pc_genes))

        valid_samples = self._get_valid_sample_ids()
        if not valid_samples:
            raise ValueError(
                "No samples in samples table. Load samples first."
            )
        self.logger.info("Found %d valid samples", len(valid_samples))

        # Tune DuckDB
        self.db_manager.execute("SET memory_limit = '4GB'")
        self.db_manager.execute("SET preserve_insertion_order = false")
        self.db_manager.execute("SET threads = 2")

        # Read header to discover columns
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

        # Create protein-coding gene lookup
        self.db_manager.execute(
            """
            CREATE OR REPLACE TEMP TABLE pc_gene_ids AS
            SELECT ensembl_id_versioned, gene_id
            FROM genes WHERE is_protein_coding = TRUE
            """
        )

        # Process in chunks of sample columns
        chunks = [
            sample_cols[i : i + SAMPLES_PER_CHUNK]
            for i in range(0, len(sample_cols), SAMPLES_PER_CHUNK)
        ]
        self.logger.info(
            "Processing %d samples in %d chunks of %d",
            len(sample_cols),
            len(chunks),
            SAMPLES_PER_CHUNK,
        )

        # Create staging long table
        self.db_manager.execute(
            """
            CREATE OR REPLACE TEMP TABLE expression_long (
                gene_id VARCHAR,
                sample_id VARCHAR,
                value DOUBLE
            )
            """
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
                INSERT INTO expression_long
                SELECT
                    g.gene_id,
                    u.sample_id,
                    TRY_CAST(u.value AS DOUBLE) AS value
                FROM (
                    UNPIVOT (
                        SELECT {select_cols}
                        FROM read_csv(
                            '{file_path}',
                            delim='\t',
                            header=true,
                            all_varchar=true
                        )
                    )
                    ON {quoted_chunk}
                    INTO NAME sample_id VALUE value
                ) u
                INNER JOIN pc_gene_ids g
                    ON u."{gene_col}" = g.ensembl_id_versioned
                """
            )

            chunk_count = self.db_manager.execute(
                "SELECT COUNT(*) FROM expression_long"
            ).fetchone()[0]
            self.logger.info("  Cumulative long rows: %d", chunk_count)
            total_rows = chunk_count

        self.logger.info("Total long format: %d rows", total_rows)

        # Get actual genes present
        genes_result = self.db_manager.execute(
            "SELECT DISTINCT gene_id FROM expression_long ORDER BY gene_id"
        )
        actual_genes = [row[0] for row in genes_result.fetchall()]
        warnings.append(f"Loaded {len(actual_genes)} protein-coding genes")

        # Export long table to Parquet for pivoting
        tmp_dir = Path(f"data/tmp/{table_name}")
        tmp_dir.mkdir(parents=True, exist_ok=True)
        long_parquet = tmp_dir / "long.parquet"

        self.logger.info("Writing long table to Parquet...")
        self.db_manager.execute(
            f"COPY expression_long TO '{long_parquet}' (FORMAT PARQUET)"
        )
        self.db_manager.execute("DROP TABLE IF EXISTS expression_long")
        self.db_manager.execute("DROP TABLE IF EXISTS pc_gene_ids")

        # Pivot in pandas batches → write wide Parquet
        sample_ids = self.db_manager.execute(
            f"SELECT DISTINCT sample_id "
            f"FROM read_parquet('{long_parquet}') ORDER BY sample_id"
        ).fetchall()
        sample_list = [r[0] for r in sample_ids]
        self.logger.info(
            "Pivoting %d samples to wide Parquet...",
            len(sample_list),
        )

        batch_size = 500
        wide_parts: list[pd.DataFrame] = []
        for i in range(0, len(sample_list), batch_size):
            batch_samples = sample_list[i : i + batch_size]
            placeholders = ", ".join(f"'{s}'" for s in batch_samples)

            batch_df = self.db_manager.execute(
                f"SELECT * FROM read_parquet('{long_parquet}') "
                f"WHERE sample_id IN ({placeholders})"
            ).df()

            wide = batch_df.pivot(
                index="sample_id", columns="gene_id", values="value"
            )
            wide = wide.reset_index()
            wide_parts.append(wide)

            if (i // batch_size) % 5 == 0:
                self.logger.info(
                    "  Pivoted %d / %d samples",
                    min(i + batch_size, len(sample_list)),
                    len(sample_list),
                )

        # Concatenate and write final wide Parquet
        self.logger.info("Writing final wide Parquet to %s", parquet_out)
        full_wide = pd.concat(wide_parts, ignore_index=True)
        full_wide.to_parquet(parquet_out, index=False)
        inserted = len(full_wide)
        del full_wide, wide_parts  # Free memory

        # Create DuckDB VIEW for SQL access
        # Drop table first (schema init creates it), then view
        self.db_manager.execute(f"DROP TABLE IF EXISTS {table_name}")
        self.db_manager.execute(f"DROP VIEW IF EXISTS {table_name}")
        abs_parquet = parquet_out.resolve()
        self.db_manager.execute(
            f"""
            CREATE VIEW {table_name} AS
            SELECT * FROM read_parquet('{abs_parquet}')
            """
        )

        # Cleanup temp files
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


class ExpressionCountsWideProcessor(ExpressionWideProcessor):
    """Loads log2(expected_count + 1) expression matrix."""

    def __init__(self, batch_size: int | None = None) -> None:
        super().__init__(
            dataset_name="expression_counts",
            batch_size=batch_size,
        )

    def get_table_name(self) -> str:
        return "expression_counts_wide"


class ExpressionTPMWideProcessor(ExpressionWideProcessor):
    """Loads log2(TPM + 0.001) expression matrix."""

    def __init__(self, batch_size: int | None = None) -> None:
        super().__init__(
            dataset_name="expression_tpm",
            batch_size=batch_size,
        )

    def get_table_name(self) -> str:
        return "expression_tpm_wide"
