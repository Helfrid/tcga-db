"""File cache management for downloaded TCGA/Xena datasets."""

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from ..config import get_logger, get_settings


@dataclass
class DownloadResult:
    """Record of a successfully downloaded dataset file.

    Attributes:
        dataset_name: Logical name identifying the dataset.
        filename: Bare filename on disk.
        file_path: Absolute path to the cached file.
        file_size: Size of the file in bytes.
        checksum: Hex digest of the file content (MD5 by default).
        download_time: ISO-8601 timestamp of when the download completed.
        source_url: Original URL the file was fetched from.
    """

    dataset_name: str
    filename: str
    file_path: Path
    file_size: int
    checksum: str
    download_time: str
    source_url: str


class FileManager:
    """Manage a local cache of downloaded dataset files.

    Metadata is persisted in a ``download_metadata.json`` file inside the
    cache directory so that download results survive process restarts.

    Args:
        cache_dir: Directory used for storing downloaded files and the
            metadata index.  Defaults to ``settings.xena.cache_dir``.
    """

    _METADATA_FILENAME = "download_metadata.json"

    def __init__(self, cache_dir: Path | None = None) -> None:
        settings = get_settings()
        self.cache_dir: Path = cache_dir or settings.xena.cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.logger = get_logger(self.__class__.__name__)
        self._metadata: dict[str, dict] = self._load_metadata()

    # ------------------------------------------------------------------
    # Metadata persistence
    # ------------------------------------------------------------------

    def _metadata_path(self) -> Path:
        return self.cache_dir / self._METADATA_FILENAME

    def _load_metadata(self) -> dict[str, dict]:
        """Load the cached metadata index from disk.

        Returns:
            A mapping of dataset name to serialised :class:`DownloadResult`
            fields.  Returns an empty dict when no index exists yet.
        """
        path = self._metadata_path()
        if not path.exists():
            return {}

        try:
            with path.open() as fh:
                return json.load(fh)
        except (json.JSONDecodeError, OSError) as exc:
            self.logger.warning(
                "Could not read metadata file %s: %s", path, exc
            )
            return {}

    def _save_metadata(self) -> None:
        """Persist the in-memory metadata index to disk."""
        path = self._metadata_path()
        try:
            with path.open("w") as fh:
                json.dump(self._metadata, fh, indent=2, default=str)
        except OSError as exc:
            self.logger.error(
                "Failed to write metadata file %s: %s", path, exc
            )

    # ------------------------------------------------------------------
    # Checksum
    # ------------------------------------------------------------------

    def calculate_file_checksum(
        self, file_path: Path, algorithm: str = "md5"
    ) -> str:
        """Compute a hex digest for a file by reading it in chunks.

        Args:
            file_path: Path to the file to hash.
            algorithm: Hash algorithm name accepted by :mod:`hashlib`
                (e.g. ``"md5"``, ``"sha256"``).

        Returns:
            Lowercase hex digest string.

        Raises:
            ValueError: If *algorithm* is not supported by :mod:`hashlib`.
            FileNotFoundError: If *file_path* does not exist.
        """
        try:
            hasher = hashlib.new(algorithm)
        except ValueError as exc:
            raise ValueError(
                f"Unsupported hash algorithm: {algorithm!r}"
            ) from exc

        chunk_size = 8 * 1024  # 8 KB
        with file_path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(chunk_size), b""):
                hasher.update(chunk)

        return hasher.hexdigest()

    # ------------------------------------------------------------------
    # Registration & lookup
    # ------------------------------------------------------------------

    def register_download(
        self,
        dataset_name: str,
        file_path: Path,
        source_url: str,
    ) -> DownloadResult:
        """Record a completed download in the metadata index.

        Args:
            dataset_name: Logical identifier for the dataset.
            file_path: Path to the file that was downloaded.
            source_url: URL the file was retrieved from.

        Returns:
            A :class:`DownloadResult` capturing all relevant metadata.
        """
        checksum = self.calculate_file_checksum(file_path)
        result = DownloadResult(
            dataset_name=dataset_name,
            filename=file_path.name,
            file_path=file_path,
            file_size=file_path.stat().st_size,
            checksum=checksum,
            download_time=datetime.now(UTC).isoformat(),
            source_url=source_url,
        )

        entry = asdict(result)
        entry["file_path"] = str(file_path)
        self._metadata[dataset_name] = entry
        self._save_metadata()

        self.logger.info(
            "Registered download: %s (%d bytes, %s)",
            dataset_name,
            result.file_size,
            checksum,
        )
        return result

    def is_file_cached(
        self,
        dataset_name: str,
        validate_integrity: bool = False,
    ) -> bool:
        """Check whether a dataset has already been downloaded.

        Args:
            dataset_name: Logical identifier for the dataset.
            validate_integrity: When ``True``, re-compute the file checksum
                and compare it against the stored value.

        Returns:
            ``True`` if the file is present (and passes integrity check when
            requested), ``False`` otherwise.
        """
        entry = self._metadata.get(dataset_name)
        if entry is None:
            return False

        file_path = Path(entry["file_path"])
        if not file_path.exists():
            self.logger.debug(
                "Cached path for %s no longer exists: %s",
                dataset_name,
                file_path,
            )
            return False

        if not validate_integrity:
            return True

        current_checksum = self.calculate_file_checksum(file_path)
        if current_checksum != entry["checksum"]:
            self.logger.warning(
                "Integrity check failed for %s: expected %s, got %s",
                dataset_name,
                entry["checksum"],
                current_checksum,
            )
            return False

        return True

    def get_file_path(self, dataset_name: str) -> Path | None:
        """Return the cached file path for a dataset, or ``None``.

        Args:
            dataset_name: Logical identifier for the dataset.

        Returns:
            Absolute :class:`~pathlib.Path` if the file exists in cache,
            ``None`` otherwise.
        """
        entry = self._metadata.get(dataset_name)
        if entry is None:
            return None

        file_path = Path(entry["file_path"])
        return file_path if file_path.exists() else None

    # ------------------------------------------------------------------
    # Cache stats & management
    # ------------------------------------------------------------------

    def get_cache_stats(self) -> dict:
        """Return a summary of the current cache state.

        Returns:
            A dict with keys:

            * ``cached_count`` – number of datasets in cache.
            * ``total_size_bytes`` – combined size of all cached files.
            * ``datasets`` – list of cached dataset names.
        """
        total_size = sum(
            entry.get("file_size", 0) for entry in self._metadata.values()
        )
        return {
            "cached_count": len(self._metadata),
            "total_size_bytes": total_size,
            "datasets": list(self._metadata.keys()),
        }

    def clear_cache(self, datasets: list[str] | None = None) -> None:
        """Remove cached files and their metadata entries.

        Args:
            datasets: Specific dataset names to remove.  When ``None``,
                **all** cached datasets are cleared.
        """
        targets = datasets if datasets is not None else list(self._metadata)

        for name in targets:
            entry = self._metadata.pop(name, None)
            if entry is None:
                self.logger.debug(
                    "Dataset %r not found in cache — skipping.", name
                )
                continue

            file_path = Path(entry["file_path"])
            if file_path.exists():
                try:
                    file_path.unlink()
                    self.logger.info("Removed cached file: %s", file_path)
                except OSError as exc:
                    self.logger.error(
                        "Could not delete %s: %s", file_path, exc
                    )
            else:
                self.logger.debug(
                    "File for %r already absent: %s", name, file_path
                )

        self._save_metadata()
