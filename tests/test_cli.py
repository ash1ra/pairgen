from pathlib import Path
from unittest.mock import patch

import pytest
from pytest import CaptureFixture

from pairgen.cli import main


def test_cli_input_path_not_exists(tmp_path: Path, capsys: CaptureFixture[str]) -> None:
    input_path = tmp_path / "fake_dir"

    test_args = ["pairgen", "-i", str(input_path), "-o", str(tmp_path), "-s", "4"]

    with patch("sys.argv", test_args):
        with pytest.raises(SystemExit):
            main()

    captured = capsys.readouterr()
    assert "Input path does not exist" in captured.err


def test_cli_invalid_workers(tmp_path: Path, capsys: CaptureFixture[str]) -> None:
    test_args = ["pairgen", "-i", str(tmp_path), "-o", str(tmp_path), "-s", "4", "-w", "0"]

    with patch("sys.argv", test_args):
        with pytest.raises(SystemExit):
            main()

    captured = capsys.readouterr()
    assert "Worker count must be >= 1" in captured.err
