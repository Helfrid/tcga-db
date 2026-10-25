"""Tests for database schema creation and management."""

import os
from collections.abc import Generator

import duckdb
import pytest

# Force in-memory database and suppress file logging for tests
os.environ["TCGA_DATABASE__MEMORY"] = "true"
os.environ["ENABLE_FILE_LOGGING"] = "false"
os.environ["LOG_LEVEL"] = "WARNING"


from tcga_db.config import reload_settings
from tcga_db.database.connection import (
    reset_db_manager,
)
from tcga_db.database.schema import (
    Schema,
    TableType,
    create_tables,
    drop_all_tables,
    get_current_schema_version,
)


@pytest.fixture(autouse=True)
def _reset_db() -> Generator[None]:
    """Reset the global database manager and settings for each test.

    Disconnecting an in-memory DuckDB drops all state, so each test
    starts with a fresh database.
    """
    reset_db_manager()
    reload_settings()
    yield
    reset_db_manager()


@pytest.fixture()
def schema() -> Schema:
    """Create a fresh Schema instance."""
    return Schema()


class TestSchemaDefinition:
    """Tests for schema table definitions."""

    def test_schema_has_expected_tables(self, schema: Schema) -> None:
        # Phase 1 + Phase 2 tables
        expected = {
            # system
            "schema_version",
            "data_imports",
            # metadata
            "genes",
            "patients",
            "samples",
            "survival",
            "sample_qc",
            # derived
            "tissue_mapping",
            # core data
            "expression_counts_wide",
            "expression_tpm_wide",
            "copy_number_wide",
            "somatic_mutations",
            "mutational_signatures",
        }
        assert set(schema._tables.keys()) == expected

    def test_schema_has_thirteen_tables(self, schema: Schema) -> None:
        assert len(schema._tables) == 13

    def test_system_tables(self, schema: Schema) -> None:
        system_tables = schema.get_tables_by_type(TableType.SYSTEM)
        names = {t.name for t in system_tables}
        assert names == {"schema_version", "data_imports"}

    def test_metadata_tables(self, schema: Schema) -> None:
        metadata_tables = schema.get_tables_by_type(TableType.METADATA)
        names = {t.name for t in metadata_tables}
        assert names == {
            "genes",
            "patients",
            "samples",
            "survival",
            "sample_qc",
        }

    def test_core_data_tables(self, schema: Schema) -> None:
        core_tables = schema.get_tables_by_type(TableType.CORE_DATA)
        names = {t.name for t in core_tables}
        assert names == {
            "expression_counts_wide",
            "expression_tpm_wide",
            "copy_number_wide",
            "somatic_mutations",
            "mutational_signatures",
        }

    def test_derived_tables(self, schema: Schema) -> None:
        derived_tables = schema.get_tables_by_type(TableType.DERIVED)
        names = {t.name for t in derived_tables}
        assert names == {"tissue_mapping"}

    def test_get_table_schema_returns_none_for_unknown(
        self, schema: Schema
    ) -> None:
        assert schema.get_table_schema("nonexistent") is None

    def test_get_table_schema_returns_correct_table(
        self, schema: Schema
    ) -> None:
        genes = schema.get_table_schema("genes")
        assert genes is not None
        assert genes.name == "genes"
        assert genes.table_type == TableType.METADATA


class TestCreationOrder:
    """Tests for dependency-ordered table creation."""

    def test_creation_order_has_all_tables(self, schema: Schema) -> None:
        order = schema.get_creation_order()
        assert len(order) == 13
        assert set(order) == set(schema._tables.keys())

    def test_schema_version_comes_first(self, schema: Schema) -> None:
        order = schema.get_creation_order()
        assert order[0] == "schema_version"

    def test_dependencies_come_before_dependents(self, schema: Schema) -> None:
        order = schema.get_creation_order()
        for table_name in order:
            table = schema._tables[table_name]
            idx = order.index(table_name)
            for dep in table.dependencies:
                dep_idx = order.index(dep)
                assert dep_idx < idx, (
                    f"{dep} (index {dep_idx}) must come before "
                    f"{table_name} (index {idx})"
                )

    def test_samples_comes_after_patients(self, schema: Schema) -> None:
        order = schema.get_creation_order()
        assert order.index("patients") < order.index("samples")

    def test_survival_comes_after_patients(self, schema: Schema) -> None:
        order = schema.get_creation_order()
        assert order.index("patients") < order.index("survival")

    def test_expression_tables_come_after_samples(
        self, schema: Schema
    ) -> None:
        order = schema.get_creation_order()
        samples_idx = order.index("samples")
        assert order.index("expression_counts_wide") > samples_idx
        assert order.index("expression_tpm_wide") > samples_idx


class TestTableCreation:
    """Tests for actual table creation in DuckDB."""

    def test_create_all_tables(self) -> None:
        create_tables()
        schema = Schema()
        for table_name in schema._tables:
            assert schema.db_manager.table_exists(table_name)

    def test_schema_version_is_set(self) -> None:
        create_tables()
        version = get_current_schema_version()
        assert version == Schema.CURRENT_VERSION

    def test_create_specific_tables(self) -> None:
        create_tables(["schema_version", "data_imports"])
        schema = Schema()
        assert schema.db_manager.table_exists("schema_version")
        assert schema.db_manager.table_exists("data_imports")
        assert not schema.db_manager.table_exists("genes")

    def test_create_unknown_table_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown table"):
            create_tables(["nonexistent"])

    def test_drop_all_tables(self) -> None:
        create_tables()
        drop_all_tables()
        schema = Schema()
        for table_name in schema._tables:
            assert not schema.db_manager.table_exists(table_name)

    def test_idempotent_creation(self) -> None:
        """Creating tables twice should not raise."""
        create_tables()
        create_tables()
        version = get_current_schema_version()
        assert version == Schema.CURRENT_VERSION


class TestGenesTable:
    """Tests for the genes table structure."""

    def test_genes_columns(self) -> None:
        create_tables()
        schema = Schema()
        info = schema.db_manager.get_table_info("genes")
        col_names = [row[1] for row in info]
        assert "gene_id" in col_names
        assert "ensembl_id_versioned" in col_names
        assert "gene_symbol" in col_names
        assert "gene_type" in col_names
        assert "chromosome" in col_names
        assert "is_protein_coding" in col_names

    def test_genes_insert_and_protein_coding_flag(self) -> None:
        create_tables()
        schema = Schema()
        db = schema.db_manager
        db.execute(
            """
            INSERT INTO genes (
                gene_id, ensembl_id_versioned, gene_symbol,
                gene_type, chromosome, chrom_start, chrom_end,
                strand, is_protein_coding
            ) VALUES
            ('ENSG00000012048', 'ENSG00000012048.10', 'BRCA1',
             'protein_coding', 'chr17', 43044295, 43170245, '-', TRUE),
            ('ENSG00000139618', 'ENSG00000139618.14', 'BRCA2',
             'protein_coding', 'chr13', 32315086, 32400266, '+', TRUE),
            ('ENSG00000225880', 'ENSG00000225880.5', 'LINC00115',
             'lincRNA', 'chr1', 761586, 762902, '-', FALSE)
            """
        )

        result = db.execute(
            "SELECT gene_symbol, is_protein_coding FROM genes "
            "ORDER BY gene_symbol"
        ).fetchall()
        assert result == [
            ("BRCA1", True),
            ("BRCA2", True),
            ("LINC00115", False),
        ]


class TestSamplesTable:
    """Tests for the samples table structure."""

    def test_samples_foreign_key_to_patients(self) -> None:
        create_tables()
        schema = Schema()
        db = schema.db_manager

        # Insert a patient first
        db.execute(
            "INSERT INTO patients (patient_id, project_id, cancer_type) "
            "VALUES ('TCGA-A7-A13E', 'TCGA-BRCA', 'BRCA')"
        )

        # Insert a sample linked to that patient
        db.execute(
            """
            INSERT INTO samples (
                sample_id, patient_id, source_study,
                primary_site, primary_site_normalised,
                sample_type, sample_type_code, is_tumour
            ) VALUES (
                'TCGA-A7-A13E-01A', 'TCGA-A7-A13E', 'TCGA',
                'Breast', 'breast',
                'Primary Tumor', '01', TRUE
            )
            """
        )

        result = db.execute(
            "SELECT sample_id, patient_id, is_tumour FROM samples"
        ).fetchone()
        assert result == ("TCGA-A7-A13E-01A", "TCGA-A7-A13E", True)

    def test_gtex_sample_with_null_patient(self) -> None:
        create_tables()
        schema = Schema()
        db = schema.db_manager

        db.execute(
            """
            INSERT INTO samples (
                sample_id, patient_id, source_study,
                primary_site, primary_site_normalised,
                sample_type, sample_type_code, is_tumour
            ) VALUES (
                'GTEX-1117F-0226-SM-5GZZ7', NULL, 'GTEX',
                'Breast', 'breast',
                'Normal', NULL, FALSE
            )
            """
        )

        result = db.execute(
            "SELECT sample_id, patient_id, source_study FROM samples"
        ).fetchone()
        assert result == (
            "GTEX-1117F-0226-SM-5GZZ7",
            None,
            "GTEX",
        )


class TestSurvivalTable:
    """Tests for the survival table structure."""

    def test_survival_endpoints(self) -> None:
        create_tables()
        schema = Schema()
        db = schema.db_manager

        db.execute(
            "INSERT INTO patients (patient_id, project_id, cancer_type) "
            "VALUES ('TCGA-A7-A13E', 'TCGA-BRCA', 'BRCA')"
        )

        db.execute(
            """
            INSERT INTO survival (
                patient_id, os_time, os_event,
                dss_time, dss_event,
                dfi_time, dfi_event,
                pfi_time, pfi_event
            ) VALUES (
                'TCGA-A7-A13E', 1500.0, FALSE,
                1500.0, FALSE,
                NULL, NULL,
                800.0, TRUE
            )
            """
        )

        result = db.execute(
            "SELECT os_time, os_event, dfi_time, pfi_event FROM survival"
        ).fetchone()
        assert result == (1500.0, False, None, True)


class TestTissueMappingTable:
    """Tests for the tissue_mapping table structure."""

    def test_tissue_mapping_composite_key(self) -> None:
        create_tables()
        schema = Schema()
        db = schema.db_manager

        db.execute(
            """
            INSERT INTO tissue_mapping (
                tcga_cancer_type, gtex_tissue,
                primary_site_normalised, description
            ) VALUES
            ('BRCA', 'Breast - Mammary Tissue', 'breast',
             'Breast cancer to mammary tissue'),
            ('LUAD', 'Lung', 'lung',
             'Lung adenocarcinoma to lung tissue'),
            ('LUSC', 'Lung', 'lung',
             'Lung squamous cell carcinoma to lung tissue')
            """
        )

        result = db.execute("SELECT COUNT(*) FROM tissue_mapping").fetchone()
        assert result == (3,)

    def test_tissue_mapping_duplicate_raises(self) -> None:
        create_tables()
        schema = Schema()
        db = schema.db_manager

        db.execute(
            """
            INSERT INTO tissue_mapping (
                tcga_cancer_type, gtex_tissue,
                primary_site_normalised
            ) VALUES ('BRCA', 'Breast - Mammary Tissue', 'breast')
            """
        )

        with pytest.raises(duckdb.ConstraintException):
            db.execute(
                """
                INSERT INTO tissue_mapping (
                    tcga_cancer_type, gtex_tissue,
                    primary_site_normalised
                ) VALUES ('BRCA', 'Breast - Mammary Tissue', 'breast')
                """
            )
