from __future__ import annotations

import functools
import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

import torch

T = TypeVar("T")


def _collect_tensors(payload: Any) -> list[torch.Tensor]:
    tensors: list[torch.Tensor] = []
    if isinstance(payload, torch.Tensor):
        return [payload]
    if isinstance(payload, dict):
        for value in payload.values():
            tensors.extend(_collect_tensors(value))
        return tensors
    if isinstance(payload, (list, tuple)):
        for value in payload:
            tensors.extend(_collect_tensors(value))
        return tensors
    slots = getattr(payload, "__slots__", None)
    if slots:
        for slot in slots:
            if hasattr(payload, slot):
                tensors.extend(_collect_tensors(getattr(payload, slot)))
        return tensors
    if hasattr(payload, "__dict__"):
        for value in vars(payload).values():
            tensors.extend(_collect_tensors(value))
    return tensors


def logged_call(name: str | None = None) -> Callable[[Callable[..., T]], Callable[..., T]]:
    def decorator(fn: Callable[..., T]) -> Callable[..., T]:
        logger = logging.getLogger(name or fn.__module__)

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            logger.debug("Calling %s", fn.__qualname__)
            return fn(*args, **kwargs)

        return wrapper

    return decorator


def ensure_directory(path_getter: Callable[..., str | Path]) -> Callable[[Callable[..., T]], Callable[..., T]]:
    def decorator(fn: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            target = Path(path_getter(*args, **kwargs))
            target.mkdir(parents=True, exist_ok=True)
            return fn(*args, **kwargs)

        return wrapper

    return decorator


def timed(metric_name: str) -> Callable[[Callable[..., T]], Callable[..., T]]:
    def decorator(fn: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            start = time.perf_counter()
            result = fn(*args, **kwargs)
            elapsed = time.perf_counter() - start
            metrics = getattr(args[0], "runtime_metrics", None) if args else None
            if isinstance(metrics, dict):
                metrics[metric_name] = elapsed
            return result

        return wrapper

    return decorator


def collect_metrics(name: str) -> Callable[[Callable[..., T]], Callable[..., T]]:
    def decorator(fn: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            owner = args[0] if args else None
            collection = getattr(owner, "runtime_metrics", None)
            if isinstance(collection, dict):
                collection.setdefault(name, 0.0)
            return fn(*args, **kwargs)

        return wrapper

    return decorator


def torch_no_grad(fn: Callable[..., T]) -> Callable[..., T]:
    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> T:
        with torch.no_grad():
            return fn(*args, **kwargs)

    return wrapper


def reset_runtime_metrics(fn: Callable[..., T]) -> Callable[..., T]:
    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> T:
        owner = args[0] if args else None
        metrics = getattr(owner, "runtime_metrics", None)
        if isinstance(metrics, dict):
            metrics.clear()
        return fn(*args, **kwargs)

    return wrapper


def capture_peak_memory(fn: Callable[..., T]) -> Callable[..., T]:
    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> T:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        result = fn(*args, **kwargs)
        metrics = getattr(args[0], "runtime_metrics", None) if args else None
        if torch.cuda.is_available() and isinstance(metrics, dict):
            metrics["peak_gpu_memory_mb"] = torch.cuda.max_memory_allocated() / (1024**2)
        return result

    return wrapper


def validate_tensor_output(fn: Callable[..., T]) -> Callable[..., T]:
    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> T:
        output = fn(*args, **kwargs)
        tensors = _collect_tensors(output)
        for tensor in tensors:
            if not torch.isfinite(tensor).all():
                raise FloatingPointError(f"{fn.__qualname__} produced non-finite values.")
        return output

    return wrapper
