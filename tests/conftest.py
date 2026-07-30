"""Shared pytest fixtures for the symmer test suite."""
import pytest
from symmer import process


@pytest.fixture(scope="session", autouse=True)
def _default_single_thread():
    """Default to single-thread processing to avoid fork/DeprecationWarning spam."""
    previous = process.method
    process.method = "single_thread"
    yield
    process.method = previous
