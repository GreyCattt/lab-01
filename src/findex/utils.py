import time
from functools import wraps

def timed(func):
    """Декоратор для заміру часу виконання функції."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        print(f"[TIMED] {func.__name__} executed in {elapsed:.4f}s")
        return result
    return wrapper