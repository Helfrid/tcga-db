"""Base ETL processor abstractions for tcga-db data ingestion."""

import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from ..config import get_logger, get_settings
from ..database.connection import get_db_manager


@dataclass
class ProcessingResult:
    """Result summary from a single file-processing operation.

    Attributes:
        processor_name: Class name of the processor that ran.
        dataset_name: Logical dataset identifier (e.g. ``"tcga_expression"``).
        source_file: Path to the file that was processed.
        records_processed: Total rows read from the source file.
        records_inserted: Rows successfully written to the database.
        records_skipped: Rows dropped during validation or deduplication.
        processing_time_seconds: Wall-clock seconds taken by the operation.
        status: One of ``"success"``, ``"partial"``, ``"failed"``,
            or ``"skipped"``.
        error_message: Human-readable error description when status is
            ``"failed"``.  ``None`` otherwise.
        warnings: Non-fatal issues collected during validation.
    """

    processor_name: str
    dataset_name: str
    source_file: Path
    records_processed: int
    records_inserted: int
    records_skipped: int
    processing_time_seconds: float
    status: str
    error_message: str | None = None
    warnings: list[str] | None = None


class BaseProcessor(ABC):
    """Abstract base class for all TCGA data ETL processors.

    Subclasses must implement :meth:`get_table_name`,
    :meth:`validate_data`, and :meth:`transform_data`.  The concrete
    orchestration logic lives in :meth:`process_file`.

    Args:
        dataset_name: Logical name for the dataset being processed.
        batch_size: Number of rows per INSERT batch.  Defaults to the
            value configured in ``settings.xena.batch_size``.

    Example:
        >>> class MyProcessor(BaseProcessor):
        ...     def get_table_name(self) -> str:
        ...         return "my_table"
        ...
        ...     def validate_data(
        ...         self, df: pd.DataFrame
        ...     ) -> tuple[pd.DataFrame, list[str]]:
        ...         return df, []
        ...
        ...     def transform_data(self, df: pd.DataFrame) -> pd.DataFrame:
        ...         return df
    """

    def __init__(
        self,
        dataset_name: str,
        batch_size: int | None = None,
    ) -> None:
        """Initialise processor with dataset name and optional batch size.

        Args:
            dataset_name: Logical identifier for the dataset.
            batch_size: INSERT batch size override.  Falls back to
                ``settings.xena.batch_size`` when ``None``.
        """
        self.dataset_name = dataset_name
        self.settings = get_settings()
        self.logger = get_logger(self.__class__.__name__)
        self.db_manager = get_db_manager()
        self.batch_size = batch_size or self.settings.xena.batch_size

    #: Source-file columns a subclass depends on.  Checked immediately
    #: after :meth:`load_file` so a renamed or absent upstream column
    #: fails loudly instead of degrading to NULL.  Subclasses that can
    #: accept any of several spellings should leave this empty and check
    #: in :meth:`validate_data`.
    EXPECTED_SOURCE_COLUMNS: tuple[str, ...] = ()

    def require_source_columns(self, df: pd.DataFrame) -> None:
        """Fail if any :attr:`EXPECTED_SOURCE_COLUMNS` is missing.

        Args:
            df: Raw DataFrame straight from :meth:`load_file`.

        Raises:
            ValueError: If one or more expected columns are absent.
        """
        missing = [
            c for c in self.EXPECTED_SOURCE_COLUMNS if c not in df.columns
        ]
        if missing:
            raise ValueError(
                f"{self.__class__.__name__}: source file is missing "
                f"expected column(s) {missing}. "
                f"Available columns: {list(df.columns)[:20]}"
            )

    # ------------------------------------------------------------------
    # Abstract interface — subclasses must override these
    # ------------------------------------------------------------------

    @abstractmethod
    def get_table_name(self) -> str:
        """Return the target database table name.

        Returns:
            Table name string used for all DML operations.
        """
        ...

    @abstractmethod
    def validate_data(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, list[str]]:
        """Validate a loaded DataFrame and return cleaned data plus warnings.

        Implementations should drop or fix invalid rows and accumulate
        human-readable warning messages for any issues found.

        Args:
            df: Raw DataFrame as returned by :meth:`load_file`.

        Returns:
            A two-tuple of ``(cleaned_df, warnings)`` where *warnings* is
            a (possibly empty) list of warning strings.
        """
        ...

    @abstractmethod
    def transform_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply domain-specific transformations to a validated DataFrame.

        Args:
            df: Validated DataFrame from :meth:`validate_data`.

        Returns:
            Transformed DataFrame ready for insertion.
        """
        ...

    # ------------------------------------------------------------------
    # Concrete helpers
    # ------------------------------------------------------------------

    def load_file(self, file_path: Path) -> pd.DataFrame:
        """Load a data file into a pandas DataFrame.

        Supported formats are inferred from the file extension:

        * ``.tsv``, ``.txt``, ``.xena``, ``.probemap`` (and their ``.gz``
          variants) — tab-separated values.  PanCanAtlas ships several
          TSV files under a ``.txt`` extension.
        * ``.csv``, ``.csv.gz`` — comma-separated values.

        Args:
            file_path: Absolute path to the source data file.

        Returns:
            DataFrame containing the file contents.

        Raises:
            FileNotFoundError: If *file_path* does not exist.
            ValueError: If the file extension is not recognised.
        """
        if not file_path.exists():
            raise FileNotFoundError(f"Source file not found: {file_path}")

        # Strip .gz suffix for extension detection
        stem_path = (
            Path(file_path.stem) if file_path.suffix == ".gz" else file_path
        )
        suffix = stem_path.suffix.lower()

        self.logger.info("Loading file: %s", file_path)

        if suffix in {".tsv", ".txt", ".xena", ".probemap"}:
            df = pd.read_csv(file_path, sep="\t")
        elif suffix == ".csv":
            df = pd.read_csv(file_path, sep=",")
        else:
            raise ValueError(
                f"Unsupported file extension '{suffix}' for {file_path}. "
                "Expected .csv, .tsv, .txt, .xena or .probemap "
                "(optionally .gz compressed)."
            )

        self.logger.info(
            "Loaded %d rows × %d columns from %s",
            len(df),
            len(df.columns),
            file_path.name,
        )
        return df

    def get_existing_record_count(self) -> int:
        """Return the number of rows currently in the target table.

        Returns:
            Row count, or ``0`` if the table does not exist yet.
        """
        table_name = self.get_table_name()

        if not self.db_manager.table_exists(table_name):
            return 0

        result = self.db_manager.execute(
            f"SELECT COUNT(*) FROM {table_name}"  # noqa: S608
        )
        row = result.fetchone()
        return int(row[0]) if row else 0

    def insert_batch(self, df: pd.DataFrame) -> tuple[int, int]:
        """INSERT OR REPLACE a batch of rows into the target table.

        Builds a parameterised ``INSERT OR REPLACE`` statement from
        the DataFrame columns and calls :meth:`execute_many` on the
        database manager.

        Args:
            df: DataFrame whose columns map directly to table columns.

        Returns:
            A two-tuple of ``(inserted, skipped)`` counts.  Currently
            *skipped* is always ``0``; subclasses may override this
            method to report per-batch skips.

        Raises:
            RuntimeError: If the INSERT fails for any batch row.
        """
        if df.empty:
            return 0, 0

        table_name = self.get_table_name()
        columns = list(df.columns)
        placeholders = ", ".join("?" * len(columns))
        col_list = ", ".join(columns)
        query = (
            f"INSERT INTO {table_name} ({col_list}) VALUES ({placeholders})"
        )

        # Convert DataFrame to native Python types for DuckDB
        # (numpy types like int64/float64 aren't handled by executemany)
        df_native = df.astype(object).where(df.notna(), other=None)
        rows: list[Any] = [
            list(row) for row in df_native.itertuples(index=False, name=None)
        ]

        self.db_manager.execute_many(query, rows)

        return len(df), 0

    def process_file(
        self,
        file_path: Path,
        force_reload: bool = False,
    ) -> ProcessingResult:
        """Orchestrate the full ETL pipeline for a single source file.

        Steps:

        1. Check for existing records — skip if data is present and
           *force_reload* is ``False``.
        2. Load the file with :meth:`load_file`.
        3. Validate with :meth:`validate_data`, collecting warnings.
        4. Transform with :meth:`transform_data`.
        5. Insert rows in batches of :attr:`batch_size`.
        6. Record the import in the ``data_imports`` audit table.
        7. Return a :class:`ProcessingResult` with timing and counts.

        Any uncaught exception transitions the result to ``"failed"``
        status and is recorded in the audit table before being
        re-raised as a ``RuntimeError``.

        Args:
            file_path: Path to the data file to process.
            force_reload: When ``True``, skip the existing-records check
                and re-insert regardless of current table state.

        Returns:
            :class:`ProcessingResult` summarising the operation.
        """
        start_time = time.perf_counter()
        warnings: list[str] = []

        try:
            # Step 1 — short-circuit if data already loaded
            existing = self.get_existing_record_count()
            if existing > 0 and not force_reload:
                self.logger.info(
                    "Skipping %s — %d records already present "
                    "(use force_reload=True to overwrite)",
                    self.dataset_name,
                    existing,
                )
                return ProcessingResult(
                    processor_name=self.__class__.__name__,
                    dataset_name=self.dataset_name,
                    source_file=file_path,
                    records_processed=0,
                    records_inserted=0,
                    records_skipped=0,
                    processing_time_seconds=time.perf_counter() - start_time,
                    status="skipped",
                )
            if existing > 0 and force_reload:
                self.clear_table()

            # Step 2 — load
            df = self.load_file(file_path)
            self.require_source_columns(df)
            records_processed = len(df)

            # Step 3 — validate
            df, validation_warnings = self.validate_data(df)
            warnings.extend(validation_warnings)
            records_skipped = records_processed - len(df)

            # Step 4 — transform
            df = self.transform_data(df)

            # Step 5 — insert in batches
            records_inserted = 0
            total_rows = len(df)
            for batch_start in range(0, total_rows, self.batch_size):
                batch = df.iloc[batch_start : batch_start + self.batch_size]
                inserted, _ = self.insert_batch(batch)
                records_inserted += inserted
                self.logger.debug(
                    "Inserted batch %d–%d (%d rows)",
                    batch_start,
                    min(batch_start + self.batch_size, total_rows),
                    inserted,
                )

            elapsed = time.perf_counter() - start_time
            status = "partial" if records_skipped > 0 else "success"

            # Step 6 — audit record
            self._record_import(
                file_path=file_path,
                records=records_inserted,
                status=status,
            )

            self.logger.info(
                "Processed %s: %d inserted, %d skipped in %.2fs",
                self.dataset_name,
                records_inserted,
                records_skipped,
                elapsed,
            )

            # Step 7 — return result
            return ProcessingResult(
                processor_name=self.__class__.__name__,
                dataset_name=self.dataset_name,
                source_file=file_path,
                records_processed=records_processed,
                records_inserted=records_inserted,
                records_skipped=records_skipped,
                processing_time_seconds=elapsed,
                status=status,
                warnings=warnings if warnings else None,
            )

        except Exception as exc:  # noqa: BLE001
            elapsed = time.perf_counter() - start_time
            error_msg = str(exc)
            self.logger.error(
                "Failed to process %s: %s", self.dataset_name, error_msg
            )
            self._record_import(
                file_path=file_path,
                records=0,
                status="failed",
                error_msg=error_msg,
            )
            raise RuntimeError(
                f"ETL pipeline failed for {self.dataset_name}: {error_msg}"
            ) from exc

    def _record_import(
        self,
        file_path: Path,
        records: int,
        status: str,
        error_msg: str | None = None,
    ) -> None:
        """Insert an audit row into the ``data_imports`` system table.

        Args:
            file_path: Source file that was processed.
            records: Number of rows successfully imported.
            status: Final status string (``"success"``, ``"partial"``,
                ``"failed"``, or ``"skipped"``).
            error_msg: Error description for failed imports; ``None``
                otherwise.
        """
        import_id = str(uuid.uuid4())
        now = datetime.now(tz=UTC)

        try:
            self.db_manager.execute(
                """
                INSERT INTO data_imports (
                    import_id,
                    import_type,
                    source_file,
                    records_imported,
                    status,
                    started_at,
                    completed_at,
                    error_message
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    import_id,
                    self.dataset_name,
                    str(file_path),
                    records,
                    status,
                    now,
                    now,
                    error_msg,
                ],
            )
            self.logger.debug(
                "Recorded import %s with status '%s'", import_id, status
            )
        except Exception as exc:  # noqa: BLE001
            # Audit failure must never mask the primary result
            self.logger.warning("Failed to record import audit entry: %s", exc)

    def clear_table(self) -> None:
        """Delete all rows from the target table.

        This is a destructive operation intended for development and
        testing workflows.  Use :meth:`process_file` with
        ``force_reload=True`` in production instead.
        """
        table_name = self.get_table_name()
        self.logger.warning("Clearing all rows from table: %s", table_name)
        self.db_manager.execute(f"DELETE FROM {table_name}")  # noqa: S608
        self.logger.info("Table %s cleared successfully", table_name)
