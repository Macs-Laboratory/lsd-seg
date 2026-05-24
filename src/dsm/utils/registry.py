from __future__ import annotations

from collections.abc import Callable
from typing import Any, Generic, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    def __init__(self, name: str) -> None:
        self.name = name
        self._items: dict[str, T] = {}

    def register(self, key: str) -> Callable[[T], T]:
        def decorator(item: T) -> T:
            if key in self._items:
                raise KeyError(f"{key} is already registered in {self.name}.")
            self._items[key] = item
            return item

        return decorator

    def get(self, key: str) -> T:
        if key not in self._items:
            raise KeyError(f"{key} is not registered in {self.name}.")
        return self._items[key]

    def build(self, key: str, *args: Any, **kwargs: Any) -> Any:
        target = self.get(key)
        return target(*args, **kwargs)

    def keys(self) -> list[str]:
        return sorted(self._items)
