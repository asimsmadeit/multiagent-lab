"""Tests for the time-limited ``concordia_mini`` compatibility surface."""

from __future__ import annotations

import subprocess
import sys


def test_concordia_mini_import_emits_actionable_deprecation_warning() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-W",
            "always::DeprecationWarning",
            "-c",
            "import concordia_mini",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "concordia_mini is deprecated" in result.stderr
    assert "import from the pinned `concordia` package" in result.stderr
