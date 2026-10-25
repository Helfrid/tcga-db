"""Database schema management for TCGA/GTEx data."""

from dataclasses import dataclass
from enum import Enum

from ..config import get_logger
from .connection import get_db_manager

logger = get_logger(__name__)


class TableType(Enum):
    """Types of tables in the schema."""

    SYSTEM = "system"
    METADATA = "metadata"
    CORE_DATA = "core_data"
    DERIVED = "derived"


@dataclass
class TableSchema:
    """Schema definition for a database table."""

    name: str
    sql: str
    table_type: TableType
    dependencies: list[str]
    description: str


class Schema:
    """Database schema management (Phase 1 + Phase 2)."""

    CURRENT_VERSION = "2.1.0"

    def __init__(self) -> None:
        self.db_manager = get_db_manager()
        self._tables: dict[str, TableSchema] = {}
        self._define_schema()

    def _define_schema(self) -> None:
        """Define the complete Phase 1 + Phase 2 database schema."""
        self._define_system_tables()
        self._define_metadata_tables()
        self._define_core_data_tables()
        self._define_derived_tables()

    # ------------------------------------------------------------------
    # System tables
    # ------------------------------------------------------------------

    def _define_system_tables(self) -> None:
        """Define system tables for schema tracking and audit."""
        self._add_table(
            TableSchema(
                name="schema_version",
                table_type=TableType.SYSTEM,
                dependencies=[],
                description="Tracks database schema version",
                sql="""
            CREATE TABLE IF NOT EXISTS schema_version (
                version VARCHAR PRIMARY KEY,
                applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                description TEXT
            )
            """,
            )
        )

        self._add_table(
            TableSchema(
                name="data_imports",
                table_type=TableType.SYSTEM,
                dependencies=["schema_version"],
                description="Track data import operations",
                sql="""
            CREATE TABLE IF NOT EXISTS data_imports (
                import_id VARCHAR PRIMARY KEY,
                import_type VARCHAR NOT NULL,
                source_file VARCHAR NOT NULL,
                file_checksum VARCHAR,
                records_imported INTEGER,
                status VARCHAR NOT NULL,
                started_at TIMESTAMP NOT NULL,
                completed_at TIMESTAMP,
                error_message TEXT,
                metadata JSON
            )
            """,
            )
        )

    # ------------------------------------------------------------------
    # Metadata tables
    # ------------------------------------------------------------------

    def _define_metadata_tables(self) -> None:
        """Define metadata tables for genes, samples, patients, survival."""
        self._add_table(
            TableSchema(
                name="genes",
                table_type=TableType.METADATA,
                dependencies=["schema_version"],
                description=(
                    "Gene metadata from GENCODE v23 probemap "
                    "(all ~60K genes, protein-coding flag for filtering)"
                ),
                sql="""
            CREATE TABLE IF NOT EXISTS genes (
                gene_id VARCHAR PRIMARY KEY,
                ensembl_id_versioned VARCHAR NOT NULL UNIQUE,
                gene_symbol VARCHAR NOT NULL,
                gene_type VARCHAR NOT NULL,
                chromosome VARCHAR,
                chrom_start BIGINT,
                chrom_end BIGINT,
                strand VARCHAR(1),
                is_protein_coding BOOLEAN NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """,
            )
        )

        self._add_table(
            TableSchema(
                name="patients",
                table_type=TableType.METADATA,
                dependencies=["schema_version"],
                description=(
                    "TCGA patients extracted from barcodes "
                    "(join target for survival data)"
                ),
                sql="""
            CREATE TABLE IF NOT EXISTS patients (
                patient_id VARCHAR PRIMARY KEY,
                project_id VARCHAR NOT NULL,
                cancer_type VARCHAR NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """,
            )
        )

        self._add_table(
            TableSchema(
                name="samples",
                table_type=TableType.METADATA,
                dependencies=["patients"],
                description=(
                    "Sample phenotype annotations from Toil phenotype file "
                    "(TCGA + GTEx, central linking table)"
                ),
                sql="""
            CREATE TABLE IF NOT EXISTS samples (
                sample_id VARCHAR PRIMARY KEY,
                patient_id VARCHAR,
                source_study VARCHAR NOT NULL,
                primary_site VARCHAR,
                primary_site_normalised VARCHAR,
                detailed_category VARCHAR,
                sample_type VARCHAR,
                sample_type_code VARCHAR(2),
                is_tumour BOOLEAN NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
            )
            """,
            )
        )

        self._add_table(
            TableSchema(
                name="survival",
                table_type=TableType.METADATA,
                dependencies=["patients"],
                description=(
                    "TCGA survival endpoints (OS, DSS, DFI, PFI) "
                    "from Liu et al. 2018"
                ),
                sql="""
            CREATE TABLE IF NOT EXISTS survival (
                patient_id VARCHAR PRIMARY KEY,
                os_time DOUBLE,
                os_event BOOLEAN,
                dss_time DOUBLE,
                dss_event BOOLEAN,
                dfi_time DOUBLE,
                dfi_event BOOLEAN,
                pfi_time DOUBLE,
                pfi_event BOOLEAN,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
            )
            """,
            )
        )

        self._add_table(
            TableSchema(
                name="sample_qc",
                table_type=TableType.METADATA,
                dependencies=["samples"],
                description=(
                    "Tumour purity, ploidy and MSI status from ABSOLUTE/ASCAT "
                    "estimates (TCGA tumour samples)"
                ),
                sql="""
            CREATE TABLE IF NOT EXISTS sample_qc (
                sample_id VARCHAR PRIMARY KEY,
                tumour_purity DOUBLE,
                ploidy DOUBLE,
                genome_doublings INTEGER,
                msi_status VARCHAR,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (sample_id) REFERENCES samples(sample_id)
            )
            """,
            )
        )

    # ------------------------------------------------------------------
    # Core data tables (wide-format expression/CN matrices, mutations)
    # ------------------------------------------------------------------

    def _define_core_data_tables(self) -> None:
        """Define wide-format expression/copy-number matrices and mutation tables."""
        self._add_table(
            TableSchema(
                name="copy_number_wide",
                table_type=TableType.CORE_DATA,
                dependencies=["samples"],
                description=(
                    "GISTIC2 thresholded copy number (TCGA tumour samples "
                    "× protein-coding genes, values -2 to +2)"
                ),
                sql="""
            CREATE TABLE IF NOT EXISTS copy_number_wide (
                sample_id VARCHAR PRIMARY KEY,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (sample_id) REFERENCES samples(sample_id)
            )
            """,
            )
        )

        self._add_table(
            TableSchema(
                name="somatic_mutations",
                table_type=TableType.CORE_DATA,
                dependencies=["samples", "patients", "genes"],
                description=(
                    "MC3 somatic mutations (SNVs + indels) with functional "
                    "annotation from Xena Pan-Cancer Atlas"
                ),
                sql="""
            CREATE TABLE IF NOT EXISTS somatic_mutations (
                mutation_id VARCHAR PRIMARY KEY,
                sample_id VARCHAR NOT NULL,
                patient_id VARCHAR NOT NULL,
                gene_id VARCHAR,
                gene_symbol VARCHAR NOT NULL,
                chromosome VARCHAR,
                start_position BIGINT,
                end_position BIGINT,
                variant_classification VARCHAR,
                variant_type VARCHAR,
                reference_allele VARCHAR,
                tumor_allele VARCHAR,
                hgvsp_short VARCHAR,
                vaf DOUBLE,
                sift VARCHAR,
                polyphen VARCHAR,
                FOREIGN KEY (sample_id) REFERENCES samples(sample_id),
                FOREIGN KEY (patient_id) REFERENCES patients(patient_id),
                FOREIGN KEY (gene_id) REFERENCES genes(gene_id)
            )
            """,
            )
        )

        self._add_table(
            TableSchema(
                name="mutational_signatures",
                table_type=TableType.CORE_DATA,
                dependencies=["patients"],
                description=(
                    "COSMIC SBS mutational signature exposures per patient "
                    "(long format: patient × signature)"
                ),
                sql="""
            CREATE TABLE IF NOT EXISTS mutational_signatures (
                patient_id VARCHAR NOT NULL,
                signature_id VARCHAR NOT NULL,
                exposure DOUBLE NOT NULL,
                PRIMARY KEY (patient_id, signature_id),
                FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
            )
            """,
            )
        )

        self._add_table(
            TableSchema(
                name="expression_counts_wide",
                table_type=TableType.CORE_DATA,
                dependencies=["samples"],
                description=(
                    "Log2(expected_count + 1) expression matrix "
                    "(samples x protein-coding genes)"
                ),
                sql="""
            CREATE TABLE IF NOT EXISTS expression_counts_wide (
                sample_id VARCHAR PRIMARY KEY,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (sample_id) REFERENCES samples(sample_id)
            )
            """,
            )
        )

        self._add_table(
            TableSchema(
                name="expression_tpm_wide",
                table_type=TableType.CORE_DATA,
                dependencies=["samples"],
                description=(
                    "Log2(TPM + 0.001) expression matrix "
                    "(samples x protein-coding genes)"
                ),
                sql="""
            CREATE TABLE IF NOT EXISTS expression_tpm_wide (
                sample_id VARCHAR PRIMARY KEY,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (sample_id) REFERENCES samples(sample_id)
            )
            """,
            )
        )

    # ------------------------------------------------------------------
    # Derived / reference tables
    # ------------------------------------------------------------------

    def _define_derived_tables(self) -> None:
        """Define derived and curated reference tables."""
        self._add_table(
            TableSchema(
                name="tissue_mapping",
                table_type=TableType.DERIVED,
                dependencies=["schema_version"],
                description=(
                    "Curated TCGA cancer type to GTEx tissue mapping "
                    "for tumour-vs-normal comparisons (~25 rows)"
                ),
                sql="""
            CREATE TABLE IF NOT EXISTS tissue_mapping (
                tcga_cancer_type VARCHAR NOT NULL,
                gtex_tissue VARCHAR NOT NULL,
                primary_site_normalised VARCHAR NOT NULL,
                description TEXT,
                PRIMARY KEY (tcga_cancer_type, gtex_tissue)
            )
            """,
            )
        )

    # ------------------------------------------------------------------
    # Schema management methods
    # ------------------------------------------------------------------

    def _add_table(self, table_schema: TableSchema) -> None:
        """Add a table schema to the registry."""
        self._tables[table_schema.name] = table_schema

    def get_table_schema(self, table_name: str) -> TableSchema | None:
        """Get schema for a specific table."""
        return self._tables.get(table_name)

    def get_tables_by_type(self, table_type: TableType) -> list[TableSchema]:
        """Get all tables of a specific type."""
        return [t for t in self._tables.values() if t.table_type == table_type]

    def get_creation_order(self) -> list[str]:
        """Get table names in dependency order for creation.

        Returns:
            List of table names sorted so that dependencies come first.

        Raises:
            RuntimeError: If circular dependencies are detected.
        """
        ordered: list[str] = []
        remaining = list(self._tables.keys())

        while remaining:
            ready = [
                name
                for name in remaining
                if all(
                    dep in ordered for dep in self._tables[name].dependencies
                )
            ]

            if not ready:
                raise RuntimeError(
                    f"Circular dependencies detected: {remaining}"
                )

            ordered.extend(ready)
            for name in ready:
                remaining.remove(name)

        return ordered

    def create_table(self, table_name: str) -> None:
        """Create a single table.

        Args:
            table_name: Name of the table to create.

        Raises:
            ValueError: If table name is not in the schema registry.
        """
        if table_name not in self._tables:
            raise ValueError(f"Unknown table: {table_name}")

        table_schema = self._tables[table_name]
        logger.info("Creating table: %s", table_name)

        try:
            self.db_manager.execute(table_schema.sql)
            logger.info("Successfully created table: %s", table_name)
        except (OSError, RuntimeError) as e:
            logger.error("Failed to create table %s: %s", table_name, e)
            raise


#: Columns added to existing tables after their initial release.
#: ``CREATE TABLE IF NOT EXISTS`` cannot add a column to a table that
#: already exists, so these are applied separately and idempotently.
_COLUMN_MIGRATIONS: tuple[tuple[str, str, str], ...] = (
    ("somatic_mutations", "sift", "VARCHAR"),
    ("somatic_mutations", "polyphen", "VARCHAR"),
)


def apply_column_migrations() -> None:
    """Add any missing columns listed in :data:`_COLUMN_MIGRATIONS`."""
    db_manager = get_db_manager()

    for table, column, col_type in _COLUMN_MIGRATIONS:
        if not db_manager.table_exists(table):
            continue
        existing = {
            row[0]
            for row in db_manager.execute(
                f"SELECT column_name FROM information_schema.columns "
                f"WHERE table_name = '{table}'"  # noqa: S608
            ).fetchall()
        }
        if column in existing:
            continue
        logger.info("Adding column %s.%s (%s)", table, column, col_type)
        db_manager.execute(
            f"ALTER TABLE {table} ADD COLUMN {column} {col_type}"
        )


def create_tables(tables: list[str] | None = None) -> None:
    """Create database tables in dependency order.

    Args:
        tables: Optional list of specific tables to create.
            If None, creates all tables.

    Raises:
        ValueError: If an unknown table name is provided.
    """
    schema = Schema()

    if tables is None:
        table_names = schema.get_creation_order()
    else:
        for table_name in tables:
            if table_name not in schema._tables:
                raise ValueError(f"Unknown table: {table_name}")
        table_names = tables

    logger.info("Creating tables: %s", table_names)

    for table_name in table_names:
        schema.create_table(table_name)

    apply_column_migrations()

    _update_schema_version(schema.CURRENT_VERSION)

    logger.info("Database schema creation completed successfully")


def get_current_schema_version() -> str | None:
    """Get the current database schema version."""
    db_manager = get_db_manager()

    if not db_manager.table_exists("schema_version"):
        return None

    try:
        result = db_manager.execute(
            "SELECT version FROM schema_version "
            "ORDER BY applied_at DESC LIMIT 1"
        )
        row = result.fetchone()
        return row[0] if row else None
    except (OSError, RuntimeError) as e:
        logger.error("Failed to get schema version: %s", e)
        return None


def _update_schema_version(
    version: str, description: str | None = None
) -> None:
    """Update the schema version record."""
    db_manager = get_db_manager()

    description = description or f"Schema version {version}"

    db_manager.execute(
        """
        INSERT INTO schema_version (version, description)
        SELECT ?, ?
        WHERE NOT EXISTS (
            SELECT 1 FROM schema_version WHERE version = ?
        )
        """,
        [version, description, version],
    )

    logger.info("Updated schema version to %s", version)


def drop_all_tables() -> None:
    """Drop all tables in reverse dependency order."""
    db_manager = get_db_manager()
    schema = Schema()

    table_names = list(reversed(schema.get_creation_order()))

    for table_name in table_names:
        try:
            db_manager.execute(f"DROP TABLE IF EXISTS {table_name}")
            logger.info("Dropped table: %s", table_name)
        except (OSError, RuntimeError) as e:
            logger.error("Failed to drop table %s: %s", table_name, e)

    logger.info("All tables dropped successfully")
