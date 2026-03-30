"""Database connection management for DuckDB."""

import contextlib
from collections.abc import Generator
from pathlib import Path
from types import TracebackType
from typing import Any

import duckdb
import pandas as pd

from ..config import get_logger, get_settings

logger = get_logger(__name__)


class DatabaseManager:
    """Manages DuckDB database connections and operations."""

    def __init__(self, db_path: Path | None = None) -> None:
        """Initialise database manager.

        Args:
            db_path: Optional path to database file. If None, uses settings.
        """
        self.settings = get_settings()
        self.db_path = db_path or self.settings.database.path
        self._connection: duckdb.DuckDBPyConnection | None = None

    def _get_connection_string(self) -> str:
        """Get the database connection string."""
        if self.settings.database.memory:
            return ":memory:"
        return str(self.db_path)

    def connect(self) -> duckdb.DuckDBPyConnection:
        """Establish database connection."""
        if self._connection is None:
            connection_string = self._get_connection_string()
            logger.info("Connecting to database: %s", connection_string)

            if not self.settings.database.memory and self.db_path:
                self.db_path.parent.mkdir(parents=True, exist_ok=True)

            self._connection = duckdb.connect(connection_string)

            if self.settings.database.threads:
                self._connection.execute(
                    f"SET threads TO {self.settings.database.threads}"
                )

            self._connection.execute(
                f"SET max_memory = '{self.settings.database.max_memory}'"
            )

            self._connection.execute("SET enable_progress_bar = true")

            logger.info("Database connection established successfully")

        return self._connection

    def disconnect(self) -> None:
        """Close database connection."""
        if self._connection is not None:
            self._connection.close()
            self._connection = None
            logger.info("Database connection closed")

    def execute(self, query: str, parameters: list[Any] | None = None) -> Any:
        """Execute a SQL query.

        Args:
            query: SQL query string.
            parameters: Optional query parameters.

        Returns:
            Query result.
        """
        conn = self.connect()
        if parameters:
            return conn.execute(query, parameters)
        return conn.execute(query)

    def execute_many(self, query: str, parameters_list: list[Any]) -> None:
        """Execute a SQL query with multiple parameter sets.

        Args:
            query: SQL query string.
            parameters_list: List of parameter tuples.
        """
        conn = self.connect()
        conn.executemany(query, parameters_list)

    def fetch_df(
        self, query: str, parameters: list[Any] | None = None
    ) -> pd.DataFrame:
        """Execute query and return results as pandas DataFrame.

        Args:
            query: SQL query string.
            parameters: Optional query parameters.

        Returns:
            DataFrame with query results.
        """
        result = self.execute(query, parameters)
        return result.df()

    def table_exists(self, table_name: str) -> bool:
        """Check if a table exists in the database.

        Args:
            table_name: Name of the table to check.

        Returns:
            True if table exists, False otherwise.
        """
        query = """
        SELECT COUNT(*) as count
        FROM information_schema.tables
        WHERE table_name = ?
        """
        result = self.execute(query, [table_name])
        return bool(result.fetchone()[0] > 0)

    def get_table_info(self, table_name: str) -> list[Any]:
        """Get information about table columns.

        Args:
            table_name: Name of the table.

        Returns:
            List of column information tuples.
        """
        if not self.table_exists(table_name):
            raise ValueError(f"Table '{table_name}' does not exist")

        result = self.execute(f"PRAGMA table_info('{table_name}')")
        return list(result.fetchall())

    def __enter__(self) -> "DatabaseManager":
        """Context manager entry."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Context manager exit."""
        self.disconnect()


# Global database manager instance
_db_manager: DatabaseManager | None = None


def get_db_manager() -> DatabaseManager:
    """Get the global database manager instance."""
    global _db_manager  # noqa: PLW0603
    if _db_manager is None:
        _db_manager = DatabaseManager()
    return _db_manager


@contextlib.contextmanager
def get_db_connection() -> Generator[duckdb.DuckDBPyConnection]:
    """Context manager for database connections.

    Yields:
        DuckDB connection.
    """
    manager = get_db_manager()
    yield manager.connect()


def reset_db_manager() -> None:
    """Reset the global database manager (useful for testing)."""
    global _db_manager  # noqa: PLW0603
    if _db_manager is not None:
        _db_manager.disconnect()
        _db_manager = None
