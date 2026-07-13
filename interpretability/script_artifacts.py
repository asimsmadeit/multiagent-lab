"""Safe activation-artifact helpers for standalone research scripts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from interpretability.data import load_activation_dataset


LEGACY_TRUST_HELP = (
    'Allow pickle-capable .pt loading or writing only for reviewed artifacts'
)


def add_legacy_trust_argument(parser: argparse.ArgumentParser) -> None:
    """Add the repository-standard explicit legacy pickle opt-in."""
    parser.add_argument(
        '--trust-legacy-pt',
        action='store_true',
        help=LEGACY_TRUST_HELP,
    )


def prefer_safe_activation_path(path: str | Path) -> Path:
    """Prefer a safe sibling manifest when a configured legacy path is used."""
    source = Path(path)
    if source.suffix == '.pt':
        safe_manifest = source.with_suffix('.json')
        if safe_manifest.exists():
            return safe_manifest
    return source


def load_activation_input(
    path: str | Path,
    *,
    trust_legacy_pt: bool = False,
) -> dict[str, Any]:
    """Load safe JSON+NPZ, or one explicitly trusted legacy activation file."""
    source = prefer_safe_activation_path(path)
    return load_activation_dataset(
        source,
        trusted_legacy=trust_legacy_pt,
    )


def download_activation_input(
    repo_id: str,
    filename: str,
    *,
    repo_type: str = 'dataset',
) -> Path:
    """Download a safe bundle when published, otherwise the requested artifact."""
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError

    requested = Path(filename)
    safe_filename = (
        requested.with_suffix('.json').as_posix()
        if requested.suffix == '.pt' else requested.as_posix()
    )
    try:
        manifest_path = Path(hf_hub_download(
            repo_id=repo_id,
            filename=safe_filename,
            repo_type=repo_type,
        ))
    except EntryNotFoundError:
        if safe_filename == requested.as_posix():
            raise
        return Path(hf_hub_download(
            repo_id=repo_id,
            filename=requested.as_posix(),
            repo_type=repo_type,
        ))

    payload = json.loads(manifest_path.read_text(encoding='utf-8'))
    array_file = payload.get('array_file')
    if not isinstance(array_file, str) or not array_file:
        raise ValueError('Activation manifest is missing its array_file reference')
    array_filename = requested.with_name(array_file).as_posix()
    hf_hub_download(
        repo_id=repo_id,
        filename=array_filename,
        repo_type=repo_type,
    )
    return manifest_path


__all__ = [
    'LEGACY_TRUST_HELP',
    'add_legacy_trust_argument',
    'download_activation_input',
    'load_activation_input',
    'prefer_safe_activation_path',
]
