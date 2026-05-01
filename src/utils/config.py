"""Lightweight YAML config loader with CLI dot-list overrides.

Intentionally avoids Hydra: a single ``OmegaConf`` merge gives us yaml loading,
nested dot access, and `key=value` CLI overrides without Hydra's directory
magic, output rerouting, or multirun complexity.

Usage in a script::

    cfg = load_config(default="configs/baseline.yaml")
    # CLI: python script.py --config cfg.yaml split.kind=scaffold_acceptor

The returned object is an ``omegaconf.DictConfig``; treat it as immutable.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf


def parse_cli(default_config: str | Path | None = None) -> tuple[Path, list[str]]:
    """Parse standard ``--config <path> [overrides...]`` CLI."""
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument(
        "--config",
        type=str,
        default=str(default_config) if default_config is not None else None,
        help="Path to base YAML config.",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Dot-list overrides, e.g. split.kind=scaffold_acceptor encoder.hidden_channels=256",
    )
    args = parser.parse_args()
    if args.config is None:
        raise SystemExit("--config is required (no default provided).")
    return Path(args.config), list(args.overrides)


def load_config(
    default: str | Path | None = None,
    extra_overrides: list[str] | None = None,
) -> DictConfig:
    """Load YAML config + apply CLI dot-list overrides.

    Args:
        default: fallback config path if --config is omitted.
        extra_overrides: additional overrides applied after CLI ones (useful
            for programmatic tweaks in tests).

    Returns:
        Merged ``DictConfig``.
    """
    config_path, cli_overrides = parse_cli(default_config=default)
    cfg = OmegaConf.load(config_path)
    if cli_overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(cli_overrides))
    if extra_overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(extra_overrides))
    return cfg  # type: ignore[return-value]


def save_config(cfg: DictConfig, path: str | Path) -> None:
    """Dump resolved config to YAML for experiment provenance."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        OmegaConf.save(cfg, f, resolve=True)


def to_dict(cfg: DictConfig) -> dict[str, Any]:
    """Convert ``DictConfig`` to a plain Python dict (for json dump etc.)."""
    return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]
