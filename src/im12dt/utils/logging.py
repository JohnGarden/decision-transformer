from __future__ import annotations
import logging
from rich.logging import RichHandler


_FMT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


_def_handler = RichHandler(rich_tracebacks=True, markup=True)


logging.basicConfig(level=logging.INFO, format=_FMT, handlers=[_def_handler])




def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)