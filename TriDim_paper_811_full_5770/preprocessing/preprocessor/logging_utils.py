from __future__ import annotations

import logging
from pathlib import Path


def configure_console_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def attach_file_logger(log_path: Path) -> logging.Handler:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(log_path, mode="w")
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
    )
    logging.getLogger().addHandler(handler)
    return handler


def detach_file_logger(handler: logging.Handler) -> None:
    logging.getLogger().removeHandler(handler)
    handler.close()
