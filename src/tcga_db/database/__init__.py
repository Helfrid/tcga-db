"""Database connection and schema management for tcga-db."""

from .connection import (
    DatabaseManager,
    get_db_connection,
    get_db_manager,
    reset_db_manager,
)
from .schema import (
    Schema,
    TableSchema,
    TableType,
    create_tables,
    drop_all_tables,
)

__all__ = [
    "DatabaseManager",
    "Schema",
    "TableSchema",
    "TableType",
    "create_tables",
    "drop_all_tables",
    "get_db_connection",
    "get_db_manager",
    "reset_db_manager",
]
