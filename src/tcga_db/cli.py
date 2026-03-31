"""CLI interface for tcga-db."""

from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from .config import get_logger, get_settings
from .database.connection import get_db_manager
from .database.schema import (
    Schema,
    create_tables,
    get_current_schema_version,
)

logger = get_logger(__name__)
console = Console()

# Load order respects foreign key dependencies
LOAD_ORDER: list[str] = [
    "gene_probemap",
    "phenotype_patients",
    "phenotype_samples",
    "survival",
    "tissue_mapping",
    "expression_counts",
    "expression_tpm",
]


def _get_processor(dataset_name: str):  # noqa: ANN202
    """Get the appropriate processor for a dataset."""
    from .etl.processors.expression_wide import (
        ExpressionCountsWideProcessor,
        ExpressionTPMWideProcessor,
    )
    from .etl.processors.gene import GeneProcessor
    from .etl.processors.patient import PatientProcessor
    from .etl.processors.sample import SampleProcessor
    from .etl.processors.survival import SurvivalProcessor
    from .etl.processors.tissue_mapping import TissueMappingProcessor

    registry: dict[str, type] = {
        "gene_probemap": GeneProcessor,
        "phenotype_patients": PatientProcessor,
        "phenotype_samples": SampleProcessor,
        "survival": SurvivalProcessor,
        "tissue_mapping": TissueMappingProcessor,
        "expression_counts": ExpressionCountsWideProcessor,
        "expression_tpm": ExpressionTPMWideProcessor,
    }
    cls = registry.get(dataset_name)
    if cls is None:
        return None

    # GeneProcessor needs the GTF path for gene biotypes
    if dataset_name == "gene_probemap":
        from .downloader.file_manager import FileManager

        fm = FileManager()
        gtf_path = fm.get_file_path("gencode_gtf")
        return GeneProcessor(gtf_path=gtf_path)

    return cls()


def _require_initialized_database() -> None:
    """Check that the database schema has been created."""
    version = get_current_schema_version()
    if version is None:
        raise click.ClickException(
            "Database not initialised. Run 'tcga-db init' first."
        )


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
@click.option("--debug", is_flag=True, help="Debug mode")
@click.pass_context
def cli(ctx: click.Context, verbose: bool, debug: bool) -> None:
    """TCGA/GTEx database CLI."""
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    ctx.obj["debug"] = debug


@cli.command()
@click.option(
    "--force",
    is_flag=True,
    help="Drop and recreate existing schema",
)
def init(force: bool) -> None:
    """Initialise the database schema."""
    settings = get_settings()
    version = get_current_schema_version()

    if version and not force:
        console.print(
            f"Database already initialised (v{version}). "
            "Use --force to reinitialise."
        )
        return

    if force and version:
        from .database.schema import drop_all_tables

        console.print("Dropping existing tables...")
        drop_all_tables()

    console.print(f"Database: {settings.database.path}")
    console.print("Creating tables...")
    create_tables()

    version = get_current_schema_version()
    console.print(f"[green]Database initialised (v{version})[/green]")

    # Show table summary
    schema = Schema()
    table = Table(title="Tables Created")
    table.add_column("Table")
    table.add_column("Type")
    for name in schema.get_creation_order():
        ts = schema.get_table_schema(name)
        if ts:
            table.add_row(name, ts.table_type.value)
    console.print(table)


@cli.command()
@click.option(
    "--datasets",
    multiple=True,
    help="Specific datasets to download (default: all Phase 1)",
)
@click.option("--force", is_flag=True, help="Re-download cached files")
@click.option(
    "--load-data",
    is_flag=True,
    help="Load data into DB after download",
)
def download(datasets: tuple[str, ...], force: bool, load_data: bool) -> None:
    """Download Xena Toil Hub data files."""
    from .downloader.xena_client import (
        EXTERNAL_SOURCES,
        PHASE_1_SOURCES,
        XenaDownloader,
    )

    all_sources = {**PHASE_1_SOURCES, **EXTERNAL_SOURCES}
    if datasets:
        requested = list(datasets)
        unknown = set(requested) - set(all_sources)
        if unknown:
            raise click.ClickException(
                f"Unknown datasets: {unknown}. Available: {list(all_sources)}"
            )
    else:
        requested = list(all_sources)

    downloader = XenaDownloader()
    downloaded = downloader.download_all(datasets=requested, force=force)

    # Show results
    table = Table(title="Download Results")
    table.add_column("Dataset")
    table.add_column("Status")
    table.add_column("File")

    for name in requested:
        if name in downloaded:
            path = downloaded[name]
            table.add_row(name, "[green]downloaded[/green]", str(path.name))
        else:
            table.add_row(name, "[red]failed[/red]", "")

    console.print(table)

    if load_data:
        _require_initialized_database()
        _load_from_cache(force=force)


@cli.command()
@click.option("--force", is_flag=True, help="Force reload existing data")
def load(force: bool) -> None:
    """Load downloaded data into the database."""
    _require_initialized_database()
    _load_from_cache(force=force)


def _load_from_cache(force: bool = False) -> None:
    """Load all cached data files into the database in order."""
    from .downloader.file_manager import FileManager

    file_manager = FileManager()
    results_table = Table(title="Load Results")
    results_table.add_column("Dataset")
    results_table.add_column("Status")
    results_table.add_column("Records")
    results_table.add_column("Time")

    for dataset_name in LOAD_ORDER:
        processor = _get_processor(dataset_name)
        if processor is None:
            continue

        # Tissue mapping doesn't need a file
        if dataset_name == "tissue_mapping":
            console.print(f"Loading {dataset_name}...")
            result = processor.process_file(force_reload=force)
            results_table.add_row(
                dataset_name,
                f"[green]{result.status}[/green]",
                str(result.records_inserted),
                f"{result.processing_time_seconds:.1f}s",
            )
            continue

        # Map processor dataset names to download dataset names
        download_name = dataset_name
        if dataset_name in ("phenotype_patients", "phenotype_samples"):
            download_name = "phenotype"

        # Get cached file
        file_path = file_manager.get_file_path(download_name)
        if file_path is None:
            console.print(
                f"[yellow]Skipping {dataset_name} — file not cached[/yellow]"
            )
            results_table.add_row(
                dataset_name, "[yellow]no file[/yellow]", "", ""
            )
            continue

        console.print(f"Loading {dataset_name} from {file_path.name}...")
        try:
            result = processor.process_file(file_path, force_reload=force)
            status_color = {
                "success": "green",
                "skipped": "yellow",
                "failed": "red",
            }.get(result.status, "white")

            results_table.add_row(
                dataset_name,
                f"[{status_color}]{result.status}[/{status_color}]",
                str(result.records_inserted),
                f"{result.processing_time_seconds:.1f}s",
            )

            if result.warnings:
                for w in result.warnings:
                    console.print(f"  [dim]{w}[/dim]")

        except Exception as e:
            logger.exception("Failed to load %s", dataset_name)
            results_table.add_row(
                dataset_name, "[red]error[/red]", "", str(e)[:50]
            )

    console.print(results_table)


@cli.command()
def status() -> None:
    """Show database status and table statistics."""
    version = get_current_schema_version()
    if version is None:
        console.print("[red]Database not initialised[/red]")
        return

    settings = get_settings()
    console.print(f"Database: {settings.database.path}")
    console.print(f"Schema version: {version}")

    db = get_db_manager()
    schema = Schema()

    table = Table(title="Table Statistics")
    table.add_column("Table")
    table.add_column("Type")
    table.add_column("Rows", justify="right")

    # Check for Parquet-backed expression views
    parquet_tables = {
        "expression_counts_wide": Path("data/expression_counts_wide.parquet"),
        "expression_tpm_wide": Path("data/expression_tpm_wide.parquet"),
    }

    for name in schema.get_creation_order():
        ts = schema.get_table_schema(name)
        if ts is None:
            continue
        if name in parquet_tables and parquet_tables[name].exists():
            count = db.execute(
                f"SELECT COUNT(*) FROM read_parquet("
                f"'{parquet_tables[name].resolve()}')"
            ).fetchone()[0]
            table.add_row(
                name, ts.table_type.value + " (parquet)", f"{count:,}"
            )
        elif db.table_exists(name):
            count = db.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0]
            table.add_row(name, ts.table_type.value, f"{count:,}")
        else:
            table.add_row(name, ts.table_type.value, "[dim]—[/dim]")

    console.print(table)


def main() -> None:
    """Entry point for the CLI."""
    cli()
