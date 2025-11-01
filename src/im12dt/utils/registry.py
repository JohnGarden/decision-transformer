from __future__ import annotations
from typing import Callable, Dict


class Registry:
    def __init__(self):
        self._f: Dict[str, Callable] = {}
    def register(self, name: str):
        def deco(fn: Callable):
            self._f[name] = fn
            return fn
            return deco
    def get(self, name: str) -> Callable:
        return self._f[name]