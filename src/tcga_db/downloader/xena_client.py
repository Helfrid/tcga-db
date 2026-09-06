"""HTTP client for downloading datasets from the Xena Toil Hub."""

from pathlib import Path

import httpx
from tqdm import tqdm

from ..config import get_logger, get_settings
from .file_manager import FileManager

PHASE_1_SOURCES: dict[str, str] = {
    "gene_probemap": "probeMap/gencode.v23.annotation.gene.probemap",
    "phenotype": "TcgaTargetGTEX_phenotype.txt.gz",
    "survival": "TCGA_survival_data",
    "expression_counts": "TcgaTargetGtex_gene_expected_count.gz",
    "expression_tpm": "TcgaTargetGtex_rsem_gene_tpm.gz",
}

# Phase 2: Xena Pan-Cancer Atlas Hub (absolute URLs, different base from Toil)
_PANCAN_BASE = "https://pancanatlas.xenahubs.net/download"

PHASE_2_SOURCES: dict[str, str] = {
    "copy_number": (
        f"{_PANCAN_BASE}"
        "/broad.mit.edu_PANCAN_Genome_Wide_SNP_6_whitelisted.gene.xena.gz"
    ),
    "somatic_mutations": (f"{_PANCAN_BASE}/mc3.v0.2.8.PUBLIC.xena.gz"),
    # ABSOLUTE purity/ploidy, served open-access by the GDC PanCanAtlas
    # publication page.  The endpoint rejects HEAD requests with 400 but
    # serves GET normally.
    "sample_qc": (
        "https://api.gdc.cancer.gov/data/4f277128-f793-4354-a13d-30cc7fe9f6b5"
    ),
}

# Sources with absolute URLs (not relative to Xena base_url)
EXTERNAL_SOURCES: dict[str, str] = {
    "gencode_gtf": (
        "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human"
        "/release_23/gencode.v23.annotation.gtf.gz"
    ),
}

# GDC data endpoints are keyed by UUID, so the URL carries no filename.
_FILENAME_OVERRIDES: dict[str, str] = {
    "sample_qc": "TCGA_mastercalls.abs_tables_JSedit.fixed.txt",
}

# Datasets with no automated download route.  PanCanAtlas publishes no
# mutational signature exposures, and the Synapse release is behind a
# login, so the file has to be fetched by hand.
MANUAL_SOURCES: dict[str, str] = {
    "mutational_signatures": (
        "COSMIC SBS signature exposures are not part of the GDC PanCanAtlas "
        "release.\n"
        "Download the TCGA SBS signature activity table from Synapse "
        "syn11738669\n"
        "(Alexandrov et al. 2020, Nature — free registration required), then:\n"
        "  1. save it as data/cache/TCGA_SBS_signatures.tsv\n"
        "  2. run: tcga-db load\n"
        "Alternatively, derive exposures locally from the MC3 mutations "
        "already\n"
        "in the database using SigProfilerExtractor or sigminer."
    ),
}

_CHUNK_SIZE = 64 * 1024  # 64 KB per read


class ManualDownloadRequired(RuntimeError):
    """Raised when a dataset cannot be fetched without manual steps.

    Attributes:
        dataset_name: The dataset that was requested.
        instructions: Human-readable steps for obtaining the file.
    """

    def __init__(self, dataset_name: str, instructions: str) -> None:
        super().__init__(
            f"{dataset_name!r} has no automated download URL.\n{instructions}"
        )
        self.dataset_name = dataset_name
        self.instructions = instructions


class XenaDownloader:
    """Download datasets from the UCSC Xena Toil Hub.

    Files are cached locally via a :class:`~.file_manager.FileManager` so
    that repeated calls do not re-download data that is already present.

    Args:
        file_manager: An optional pre-configured :class:`FileManager`.
            A new instance is created when ``None`` is passed.
    """

    def __init__(self, file_manager: FileManager | None = None) -> None:
        self.file_manager: FileManager = file_manager or FileManager()
        self.settings = get_settings()
        self.logger = get_logger(self.__class__.__name__)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def get_available_datasets(self) -> list[str]:
        """Return the list of dataset names available for download.

        Returns:
            Sorted list of keys from all source dictionaries.
        """
        return sorted(
            {
                **PHASE_1_SOURCES,
                **PHASE_2_SOURCES,
                **EXTERNAL_SOURCES,
                **MANUAL_SOURCES,
            }
        )

    def download_file(
        self,
        dataset_name: str,
        force: bool = False,
    ) -> Path:
        """Download a single dataset file and return its local path.

        If the file is already cached and *force* is ``False``, the cached
        path is returned immediately without making a network request.

        Args:
            dataset_name: Key from :data:`PHASE_1_SOURCES` identifying the
                dataset to download.
            force: When ``True``, re-download even if the file is cached.

        Returns:
            Path to the locally cached file.

        Raises:
            KeyError: If *dataset_name* is not found in
                :data:`PHASE_1_SOURCES`.
            httpx.HTTPStatusError: If the server returns a non-2xx status
                after all retries are exhausted.
            httpx.RequestError: If a network-level error persists beyond the
                retry limit.
            ManualDownloadRequired: If the dataset is listed in
                :data:`MANUAL_SOURCES` and must be fetched by hand.
        """
        if dataset_name in MANUAL_SOURCES:
            raise ManualDownloadRequired(
                dataset_name, MANUAL_SOURCES[dataset_name]
            )

        all_sources = {
            **PHASE_1_SOURCES,
            **PHASE_2_SOURCES,
            **EXTERNAL_SOURCES,
        }
        if dataset_name not in all_sources:
            raise KeyError(
                f"Unknown dataset {dataset_name!r}. "
                f"Available: {sorted(all_sources)}"
            )

        if not force and self.file_manager.is_file_cached(dataset_name):
            cached_path = self.file_manager.get_file_path(dataset_name)
            if cached_path is not None:
                self.logger.info(
                    "Using cached file for %r: %s", dataset_name, cached_path
                )
                return cached_path

        url = self._build_url(dataset_name)
        filename = _FILENAME_OVERRIDES.get(
            dataset_name, Path(all_sources[dataset_name]).name
        )
        dest_path = self.file_manager.cache_dir / filename

        self.logger.info("Downloading %r from %s", dataset_name, url)
        self._download_with_retry(url, dest_path, dataset_name)

        return self.file_manager.register_download(
            dataset_name, dest_path, url
        ).file_path

    def download_all(
        self,
        datasets: list[str] | None = None,
        force: bool = False,
    ) -> dict[str, Path]:
        """Download multiple datasets and return a mapping to their paths.

        Args:
            datasets: List of dataset names to download.  Defaults to all
                keys in :data:`PHASE_1_SOURCES` when ``None``.
            force: Passed through to :meth:`download_file`.

        Returns:
            Mapping of dataset name to local file path for every dataset
            that was downloaded (or already cached) successfully.  Datasets
            that fail are omitted and their errors are logged.
        """
        all_sources = {
            **PHASE_1_SOURCES,
            **PHASE_2_SOURCES,
            **EXTERNAL_SOURCES,
        }
        targets = datasets if datasets is not None else list(all_sources)
        results: dict[str, Path] = {}

        for name in targets:
            try:
                results[name] = self.download_file(name, force=force)
            except KeyError:
                self.logger.error("Unknown dataset name: %r — skipping.", name)
            except ManualDownloadRequired as exc:
                self.logger.warning(
                    "%r requires a manual download:\n%s",
                    name,
                    exc.instructions,
                )
            except (httpx.HTTPStatusError, httpx.RequestError) as exc:
                self.logger.error(
                    "Failed to download %r after retries: %s", name, exc
                )

        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_url(self, dataset_name: str) -> str:
        """Construct the full download URL for a dataset.

        Args:
            dataset_name: Key from :data:`PHASE_1_SOURCES` or
                :data:`EXTERNAL_SOURCES`.

        Returns:
            Absolute URL string.
        """
        if dataset_name in EXTERNAL_SOURCES:
            return EXTERNAL_SOURCES[dataset_name]
        if dataset_name in PHASE_2_SOURCES:
            return PHASE_2_SOURCES[dataset_name]
        base = self.settings.xena.base_url.rstrip("/")
        return f"{base}/{PHASE_1_SOURCES[dataset_name]}"

    def _download_with_retry(
        self,
        url: str,
        dest_path: Path,
        dataset_name: str,
    ) -> None:
        """Stream a file to *dest_path*, retrying on transient failures.

        Args:
            url: Full URL to download from.
            dest_path: Local path where the file will be written.
            dataset_name: Used only for progress-bar labelling.

        Raises:
            httpx.HTTPStatusError: On a non-2xx response after all retries.
            httpx.RequestError: On a network error after all retries.
        """
        max_retries = self.settings.xena.max_retries
        timeout = self.settings.xena.timeout_seconds
        last_exc: Exception = RuntimeError("All retries exhausted")

        for attempt in range(1, max_retries + 1):
            try:
                self._stream_to_file(url, dest_path, dataset_name, timeout)
                return
            except (httpx.HTTPStatusError, httpx.RequestError) as exc:
                last_exc = exc
                self.logger.warning(
                    "Attempt %d/%d failed for %r: %s",
                    attempt,
                    max_retries,
                    dataset_name,
                    exc,
                )

        raise last_exc

    def _stream_to_file(
        self,
        url: str,
        dest_path: Path,
        dataset_name: str,
        timeout: int,
    ) -> None:
        """Perform a single streaming download with a tqdm progress bar.

        Args:
            url: Download URL.
            dest_path: Destination file path.
            dataset_name: Label used in the progress bar.
            timeout: Request timeout in seconds.

        Raises:
            httpx.HTTPStatusError: If the server returns a non-2xx status.
            httpx.RequestError: On any network-level error.
        """
        with httpx.stream(
            "GET",
            url,
            timeout=timeout,
            follow_redirects=True,
        ) as response:
            response.raise_for_status()

            content_length = response.headers.get("Content-Length")
            total_bytes = int(content_length) if content_length else None

            with (
                dest_path.open("wb") as fh,
                tqdm(
                    total=total_bytes,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    desc=dataset_name,
                    leave=False,
                ) as progress,
            ):
                for chunk in response.iter_bytes(chunk_size=_CHUNK_SIZE):
                    fh.write(chunk)
                    progress.update(len(chunk))

        self.logger.info("Saved %r to %s", dataset_name, dest_path)
