import sys
import time
from collections.abc import Callable
from functools import wraps
from typing import ParamSpec, TypeVar

P = ParamSpec('P')
R = TypeVar('R')

def timed(func: Callable[P, R]) -> Callable[P, R]:
    """Декоратор для заміру часу виконання функції."""
    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        print(f"[TIMED] {func.__name__} executed in {elapsed:.4f}s", file=sys.stderr)
        return result
    return wrapper