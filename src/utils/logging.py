import logging
import sys
from pathlib import Path


def setup_logger(name: str, log_dir: Path, level: int = logging.INFO) -> logging.Logger:
    """Configure a logger that writes to file and stdout.

    Args:
        name: Logger name (usually experiment ID or module name).
        log_dir: Directory where the log file will be written.
        level: Console log level (file always captures DEBUG).

    Returns:
        Configured Logger instance.
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    if logger.handlers:
        return logger

    fmt = logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")

    fh = logging.FileHandler(log_dir / f"{name}.log", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger
