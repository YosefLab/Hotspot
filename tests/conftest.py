import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "gpu: tests requiring a CUDA GPU and CuPy"
    )
