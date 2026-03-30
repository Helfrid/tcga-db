import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path

from dotenv import load_dotenv

# Project root of the generated app (two levels up from this config module)
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def set_env_vars() -> None:
    """Load environment variables for logging with sensible defaults.

    The configuration strategy is:

    1. Start with hard-coded defaults suitable for development.
    2. Load a base ``.env`` file if present.
    3. Load an environment-specific ``.env.<ENV>`` file (if present),
       overriding previous values.

    This matches the behaviour described in ``logging-setup.md`` and
    avoids hard failures when configuration files are missing.  It also
    keeps the API simple: calling :func:`set_env_vars` is always safe.
    """

    # Default environment and config files
    env = os.getenv("ENV", "development").lower()

    base_env_path = PROJECT_ROOT / ".env"
    env_specific_path = PROJECT_ROOT / f".env.{env}"

    # 1) Load base .env (if it exists)
    if base_env_path.exists():
        load_dotenv(base_env_path)

    # 2) Load environment-specific overrides (if they exist)
    if env_specific_path.exists():
        load_dotenv(env_specific_path, override=True)

    # 3) Ensure sane defaults for logging-related variables
    defaults: dict[str, str] = {
        "ENV": env,
        "LOG_LEVEL": "INFO",
        "LOG_FORMAT": "% (asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
        "ENABLE_CONSOLE_LOGGING": "True",
        "ENABLE_FILE_LOGGING": "True",
        "LOG_FILE_PATH": "logs/app.log",
        "LOG_MAX_BYTES": "1048576",  # 1 MB
        "LOG_BACKUP_COUNT": "5",
    }

    for key, value in defaults.items():
        os.environ.setdefault(key, value)


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

    ``name`` should normally be ``__name__`` from the calling module.
    When ``__main__`` is passed, this function attempts to resolve the
    proper module path relative to ``PROJECT_ROOT / "src"`` so that
    log records have stable, import-style names.
    """

    # Handle the case when a module is run directly (``__main__``)
    if name == "__main__":
        import inspect

        frame = inspect.stack()[1]
        module_path = Path(frame.filename)
        try:
            # Get relative path from project root's ``src`` directory
            rel_path = module_path.relative_to(PROJECT_ROOT / "src")
            # Convert path to module notation (``my_app.submodule.file``)
            module_name = str(rel_path.with_suffix("")).replace(os.sep, ".")
            name = module_name
        except ValueError:
            # Fallback if file is not in ``src`` directory
            name = module_path.stem

    # Get or create the logger
    logger = logging.getLogger(name)

    # If the root logger isn't configured yet, configure it once
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        # Make sure we have at least default environment variables
        set_env_vars()

        # Retrieve logging configuration from environment variables
        LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
        LOG_FORMAT = os.getenv(
            "LOG_FORMAT",
            "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
        )
        ENABLE_CONSOLE_LOGGING = os.getenv(
            "ENABLE_CONSOLE_LOGGING", "True"
        ).lower() in {"true", "1", "yes"}
        ENABLE_FILE_LOGGING = os.getenv(
            "ENABLE_FILE_LOGGING", "True"
        ).lower() in {"true", "1", "yes"}
        LOG_FILE_PATH = os.getenv("LOG_FILE_PATH", "logs/app.log")
        LOG_MAX_BYTES = int(os.getenv("LOG_MAX_BYTES", "1048576"))
        LOG_BACKUP_COUNT = int(os.getenv("LOG_BACKUP_COUNT", "5"))

        # Configure the root logger
        root_logger.setLevel(getattr(logging, LOG_LEVEL, logging.DEBUG))
        root_logger.propagate = False  # Do not leak to the global root

        formatter = logging.Formatter(LOG_FORMAT)

        # Console handler
        if ENABLE_CONSOLE_LOGGING:
            console_handler = logging.StreamHandler()
            configure_log_handler(
                console_handler, LOG_LEVEL, formatter, root_logger
            )

        # File handler (with rotation)
        if ENABLE_FILE_LOGGING:
            log_path = Path(LOG_FILE_PATH)
            if log_dir := log_path.parent:
                log_dir.mkdir(parents=True, exist_ok=True)

            file_handler = RotatingFileHandler(
                LOG_FILE_PATH,
                maxBytes=LOG_MAX_BYTES,
                backupCount=LOG_BACKUP_COUNT,
            )
            configure_log_handler(
                file_handler, LOG_LEVEL, formatter, root_logger
            )

    return logger
