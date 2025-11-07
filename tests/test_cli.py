"""Test CLI argument parsing."""
from footai.cli import create_parser
import pytest

def test_download_command_parsing():
    """Test that download command is recognized."""
    parser = create_parser()
    args = parser.parse_args(['download', '--country', 'EN', '--season-start', '24'])
    assert args.cmd == 'download'
    assert args.country == 'EN'
    assert args.season_start == '24'

def test_elo_multiseason_flag():
    """Test multiseason flag parsing."""
    parser = create_parser()
    args = parser.parse_args(['elo', '--season-start', '23,24', '-m', '--decay-factor', '0.9'])
    assert args.multiseason is True
    assert args.decay_factor == 0.9

def test_invalid_division_fails():
    """Test that invalid division raises error."""
    parser = create_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(['download', '--division', 'INVALID'])
