import multiprocessing
import os
def run_test():
    code = """
from typing import List

def has_close_elements(numbers: List[float], threshold: float) -> bool:
    if len(numbers) < 2:
        return False
    for i in range(len(numbers)):
        for j in range(i + 1, len(numbers)):
            if abs(numbers[i] - numbers[j]) < threshold:
                return True
    return False

METADATA = {
    'author': 'jt',
    'dataset': 'test'
}

def check(candidate):
    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3) == True
    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.05) == False

check(has_close_elements)
"""
    exec_globals = {}
    try:
        exec(code, exec_globals)
        print("passed")
    except Exception as e:
        print(f"failed: {e}")

run_test()
