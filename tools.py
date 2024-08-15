import numpy as np


def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))




import time
import functools
import os
import psutil




def timeit(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Execution time of {func.__name__}: {execution_time:.4f} seconds")
        return result
    return wrapper



def memory(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss
        result = func(*args, **kwargs)
        mem_after = process.memory_info().rss
        mem_usage = (mem_after - mem_before) / (1024 ** 2)
        print(f"Memory usage of {func.__name__}: {mem_usage:.2f} MB")
        return result
    return wrapper
