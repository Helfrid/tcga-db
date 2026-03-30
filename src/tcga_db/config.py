"""Application configuration and logging setup for tcga-db."""

import contextlib
import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Project root of the generated app (two levels up from this config module)
PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ========================
# Environment Loading
# ========================


def set_env_vars() -> None:
    """Load environment variables for logging with sensible defaults.

    The configuration strategy is:

    1. Start with hard-coded defaults suitable for development.
    2. Load a base ``.env`` file if present.
    3. Load an environment-specific ``.env.<ENV>`` file (if present),
       overriding previous values.
    """
    env = os.getenv("ENV", "development").lower()

    base_env_path = PROJECT_ROOT / ".env"
    env_specific_path = PROJECT_ROOT / f".env.{env}"

    if base_env_path.exists():
        load_dotenv(base_env_path)

    if env_specific_path.exists():
        load_dotenv(env_specific_path, override=True)

    defaults: dict[str, str] = {
        "ENV": env,
        "LOG_LEVEL": "INFO",
        "LOG_FORMAT": (
            "%(asctime)s - %(name)s - %(levelname)s"
            " - %(filename)s:%(lineno)d - %(message)s"
        ),
        "ENABLE_CONSOLE_LOGGING": "True",
        "ENABLE_FILE_LOGGING": "True",
        "LOG_FILE_PATH": "logs/app.log",
        "LOG_MAX_BYTES": "1048576",
        "LOG_BACKUP_COUNT": "5",
    }

    for key, value in defaults.items():
        os.environ.setdefault(key, value)


# ========================
# Logging
# ========================


def configure_log_handler(
    handler: logging.Handler,
    log_level: str,
    formatter: logging.Formatter,
    logger: logging.Logger,
) -> None:
    """Configure a logging handler with the specified settings.

    Args:
        handler: The logging handler to configure.
        log_level: The logging level to set.
        formatter: The formatter to use for log messages.
        logger: The logger to add the handler to.
    """
    handler.setLevel(getattr(logging, log_level, logging.DEBUG))
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger for the given module name.

    The first call configures the root logger based on environment
    variables (typically set via :func:`set_env_vars`).  Subsequent
    calls return child loggers that inherit this configuration.

    Args:
        name: The logger name, typically ``__name__`` from the calling
            module. When ``__main__`` is passed, this function resolves
            the proper module path relative to ``PROJECT_ROOT / "src"``.
    """
    if name == "__main__":
        import inspect

        frame = inspect.stack()[1]
        module_path = Path(frame.filename)
        try:
            rel_path = module_path.relative_to(PROJECT_ROOT / "src")
            module_name = str(rel_path.with_suffix("")).replace(os.sep, ".")
            name = module_name
        except ValueError:
            name = module_path.stem

    logger = logging.getLogger(name)

    root_logger = logging.getLogger()
    if not root_logger.handlers:
        set_env_vars()

        log_level = os.getenv("LOG_LEVEL", "INFO").upper()
        log_format = os.getenv(
            "LOG_FORMAT",
            "%(asctime)s - %(name)s - %(levelname)s"
            " - %(filename)s:%(lineno)d - %(message)s",
        )
        enable_console = os.getenv(
            "ENABLE_CONSOLE_LOGGING", "True"
        ).lower() in {"true", "1", "yes"}
        enable_file = os.getenv("ENABLE_FILE_LOGGING", "True").lower() in {
            "true",
            "1",
            "yes",
        }
        log_file_path = os.getenv("LOG_FILE_PATH", "logs/app.log")
        log_max_bytes = int(os.getenv("LOG_MAX_BYTES", "1048576"))
        log_backup_count = int(os.getenv("LOG_BACKUP_COUNT", "5"))

        root_logger.setLevel(getattr(logging, log_level, logging.DEBUG))
        root_logger.propagate = False

        formatter = logging.Formatter(log_format)

        if enable_console:
            console_handler = logging.StreamHandler()
            configure_log_handler(
                console_handler, log_level, formatter, root_logger
            )

        if enable_file:
            log_path = Path(log_file_path)
            if log_dir := log_path.parent:
                log_dir.mkdir(parents=True, exist_ok=True)

            file_handler = RotatingFileHandler(
                log_file_path,
                maxBytes=log_max_bytes,
                backupCount=log_backup_count,
            )
            configure_log_handler(
                file_handler, log_level, formatter, root_logger
            )

    return logger


# ========================
# Pydantic Settings Models
# ========================


class DatabaseSettings(BaseModel):
    """DuckDB database configuration."""

    path: Path = Field(
        default=Path("data/tcga.duckdb"),
        description="Path to the DuckDB database file",
    )
    memory: bool = Field(
        default=False,
        description="Use in-memory database instead of file",
    )
    threads: int | None = Field(
        default=None,
        description="Number of threads for DuckDB operations",
    )
    max_memory: str = Field(
        default="4GB",
        description="Maximum memory for DuckDB operations",
    )

    @field_validator("path")
    @classmethod
    def validate_db_path(cls, v: Path) -> Path:
        """Ensure the database directory exists."""
        if not v.parent.exists():
            v.parent.mkdir(parents=True, exist_ok=True)
        return v


class XenaSettings(BaseModel):
    """Xena Toil Hub download configuration."""

    base_url: str = Field(
        default="https://toil-xena-hub.s3.us-east-1.amazonaws.com/download",
        description="Base URL for Xena Toil Hub downloads",
    )
    cache_dir: Path = Field(
        default=Path("data/cache"),
        description="Directory for caching downloaded files",
    )
    batch_size: int = Field(
        default=10000,
        ge=1000,
        le=100000,
        description="Batch size for data processing operations",
    )
    timeout_seconds: int = Field(
        default=300,
        ge=30,
        le=3600,
        description="Request timeout in seconds",
    )
    max_retries: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum number of download retry attempts",
    )
    verify_checksums: bool = Field(
        default=True,
        description="Verify file integrity using checksums",
    )

    @field_validator("cache_dir")
    @classmethod
    def validate_cache_dir(cls, v: Path) -> Path:
        """Ensure the cache directory exists."""
        v.mkdir(parents=True, exist_ok=True)
        return v


class AppSettings(BaseSettings):
    """Main application settings combining all configurations."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        env_prefix="TCGA_",
        case_sensitive=False,
        extra="ignore",
    )

    env: str = Field(
        default="development",
        description="Application environment",
    )
    debug: bool = Field(
        default=False,
        description="Enable debug mode",
    )

    # Nested configuration models
    database: DatabaseSettings = Field(
        default_factory=DatabaseSettings,
    )
    xena: XenaSettings = Field(
        default_factory=XenaSettings,
    )

    @field_validator("env")
    @classmethod
    def validate_env(cls, v: str) -> str:
        """Normalise environment name."""
        return v.lower()


# Global settings instance
_settings: AppSettings | None = None


def get_settings() -> AppSettings:
    """Get the global application settings instance."""
    global _settings  # noqa: PLW0603
    if _settings is None:
        with contextlib.suppress(OSError):
            set_env_vars()
        _settings = AppSettings()
    return _settings


def reload_settings() -> AppSettings:
    """Force reload of settings (useful for testing)."""
    global _settings  # noqa: PLW0603
    _settings = None
    return get_settings()
