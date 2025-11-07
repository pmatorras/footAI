"""Shared pytest fixtures."""
import pytest
from pathlib import Path
import tempfile

@pytest.fixture
def temp_data_dir():
    """Temporary directory for test data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)

@pytest.fixture
def sample_args(temp_data_dir):
    """Mock arguments for testing."""
    class Args:
        country = 'SP'
        division = ['SP1']
        season_start = '24'
        raw_dir = str(temp_data_dir / 'raw')
        processed_dir = str(temp_data_dir / 'processed')
        multiseason = False
        verbose = False
        decay_factor = 0.95
    return Args()
