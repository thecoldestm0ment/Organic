from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "config.yaml"


class Config(dict):
    """Parsed config.yaml. `base_dir` is the folder the config file lives in."""

    base_dir: Path = ROOT

    def path(self, key: str) -> Path:
        p = Path(self["paths"][key])
        if not p.is_absolute():
            p = self.base_dir / p
        p.mkdir(parents=True, exist_ok=True)
        return p


def load_config(path: str | os.PathLike | None = None) -> Config:
    path = Path(path or os.environ.get("BETA_CONFIG") or DEFAULT_CONFIG)
    with open(path, encoding="utf-8") as f:
        cfg = Config(yaml.safe_load(f))
    cfg.base_dir = path.resolve().parent
    return cfg


def load_env(path: Path | None = None) -> None:
    """Minimal .env loader (KEY=VALUE per line). Variables already set win."""
    path = path or ROOT / ".env"
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def setup_console() -> None:
    """Windows consoles default to cp949; make Korean output safe."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def base_parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", default=None, help="config.yaml 경로 (기본: 프로젝트 루트)")
    return p
