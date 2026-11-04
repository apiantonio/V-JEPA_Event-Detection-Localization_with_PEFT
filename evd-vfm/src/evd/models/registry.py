"""Generic registry: adding a component does not change the orchestrators.

A new backbone is a file plus a decorator, not an ``if`` in the runner.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Generic, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._items: dict[str, T] = {}

    def register(self, name: str) -> Callable[[T], T]:
        def deco(obj: T) -> T:
            if name in self._items:
                # Registering the same name twice is almost always a double import.
                raise KeyError(f"{self.kind} '{name}' already registered by {self._items[name]!r}")
            self._items[name] = obj
            return obj

        return deco

    def get(self, name: str) -> T:
        try:
            return self._items[name]
        except KeyError:
            raise KeyError(
                f"unknown {self.kind} '{name}'. Registered: {', '.join(self.names()) or '(none)'}"
            ) from None

    def create(self, name: str, **kwargs: object) -> object:
        factory = self.get(name)
        return factory(**kwargs)  # type: ignore[operator]

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._items))

    def __contains__(self, name: object) -> bool:
        return name in self._items

    def __len__(self) -> int:
        return len(self._items)
