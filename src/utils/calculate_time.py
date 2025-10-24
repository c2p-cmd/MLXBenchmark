import time
from functools import wraps


def calculate_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        duration = time.time() - start
        print(f"Execution time for {func.__name__}: {duration:.4f}s")
        return result

    return wrapper
