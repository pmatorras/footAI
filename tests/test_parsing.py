import pytest
from footai.utils.paths import parse_start_years

def test_expand_range():
    assert parse_start_years('15-24') == ['15','16','17','18','19','20','21','22','23','24']

def test_expand_individual():
    assert parse_start_years('22,23,24') == ['22','23','24']

def test_expand_mixed():
    assert parse_start_years('15-20,23,24') == ['15','16','17','18','19','20','23','24']

def test_invalid_range():
    with pytest.raises(ValueError, match="start > end"):
        parse_start_years('24-15')
