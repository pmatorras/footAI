"""Test input validators."""
import pytest
from footai.core.validators import validate_decay_factor

def test_valid_decay_factor():
    """Test valid decay factor."""
    assert validate_decay_factor('0.95') == 0.95
    assert validate_decay_factor('0.5') == 0.5

def test_invalid_decay_factor_out_of_range():
    """Test decay factor outside 0-1 range."""
    with pytest.raises(Exception):  # argparse.ArgumentTypeError
        validate_decay_factor('1.5')
    with pytest.raises(Exception):
        validate_decay_factor('-0.1')

def test_invalid_decay_factor_not_number():
    """Test non-numeric decay factor."""
    with pytest.raises(Exception):
        validate_decay_factor('abc')
