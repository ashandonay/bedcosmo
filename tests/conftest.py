"""Pytest configuration and fixtures for plotting tests."""

import os
import sys

import pytest

# Only set environment variables if they're not already set
# This ensures tests can run without requiring actual environment setup
if 'SCRATCH' not in os.environ:
    os.environ['SCRATCH'] = '/tmp/mock_scratch'
if 'HOME' not in os.environ:
    os.environ['HOME'] = '/tmp/mock_home'

# Note: We don't mock dependencies here since they're installed in the conda environment.
# If you need to mock specific dependencies for testing, do it per-test with pytest patches.


@pytest.fixture
def mock_scratch_env(monkeypatch):
    """Override SCRATCH (and HOME if unset) for the duration of a test.
    Use this when a test needs a known SCRATCH value (e.g. /mock/scratch).
    """
    monkeypatch.setenv("SCRATCH", "/mock/scratch")
    if "HOME" not in os.environ:
        monkeypatch.setenv("HOME", "/tmp/mock_home")
