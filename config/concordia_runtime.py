"""Pinned upstream Concordia runtime metadata and validation."""

from __future__ import annotations

from importlib import metadata


UPSTREAM_DISTRIBUTION = "gdm-concordia"
UPSTREAM_VERSION = "2.4.0"
UPSTREAM_TAG = "v2.4.0"
UPSTREAM_TAG_COMMIT = "702998f57da71f87bf4e607abc1325ee51cca21f"
AUDITED_MAIN_COMMIT = "361ae4192b0701e6608963a5a968fd1e2006d3a5"


def installed_version() -> str:
    """Return the installed Concordia distribution version."""
    return metadata.version(UPSTREAM_DISTRIBUTION)


def require_supported_version() -> None:
    """Fail early when runtime code and the installed API can diverge."""
    version = installed_version()
    if version != UPSTREAM_VERSION:
        raise RuntimeError(
            f"Unsupported {UPSTREAM_DISTRIBUTION} version {version}; "
            f"expected {UPSTREAM_VERSION} ({UPSTREAM_TAG_COMMIT})."
        )
