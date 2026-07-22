"""Pytest configuration and fixtures for plotting tests."""

import os
import sys
from pathlib import Path

import pytest

# The reusable entropy estimators live in scripts/entropy_models (co-located with
# the toy-model plots that also use them). Put that dir on sys.path so tests can
# import them directly, e.g. ``from estimators import flow_plugin_entropy``.
_ENTROPY_MODELS = Path(__file__).resolve().parent.parent / "scripts" / "entropy_models"
if str(_ENTROPY_MODELS) not in sys.path:
    sys.path.insert(0, str(_ENTROPY_MODELS))

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
