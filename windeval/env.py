"""Paths and import setup for villa's winding-model code and the local `vc` stand-in.

Environment variables:
  WINDEVAL_DATA  data root (default ~/kaggle-data/vesuvius): CT mirrors, meshes, runs, temp files
  VILLA_DIR      villa checkout (default $WINDEVAL_DATA/villa)
Config files may use ${WINDEVAL_DATA}, ${VILLA_DIR} and ~ in paths.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SHIM_DIR = ROOT / "vcshim"
DATA_ROOT = Path(os.environ.get("WINDEVAL_DATA", "~/kaggle-data/vesuvius")).expanduser()
VILLA_DIR = Path(os.environ.get("VILLA_DIR", str(DATA_ROOT / "villa"))).expanduser()


def expand_path(path: str | os.PathLike) -> Path:
    """Expand ${WINDEVAL_DATA}, ${VILLA_DIR}, other env vars and ~; relative paths are repo-relative."""
    text = str(path)
    env = dict(os.environ)
    env.setdefault("WINDEVAL_DATA", str(DATA_ROOT))
    env.setdefault("VILLA_DIR", str(VILLA_DIR))
    for key in ("WINDEVAL_DATA", "VILLA_DIR"):
        text = text.replace("${" + key + "}", env[key]).replace("$" + key, env[key])
    text = os.path.expanduser(os.path.expandvars(text))
    p = Path(text)
    return p if p.is_absolute() else ROOT / p


def setup() -> None:
    os.environ.setdefault("WANDB_MODE", "disabled")
    os.environ.setdefault("WINDEVAL_DATA", str(DATA_ROOT))
    # villa's tifxyz reader caches decoded coordinates under tempfile.gettempdir();
    # keep every derived file next to the data instead of the system temp dir.
    tmp = DATA_ROOT / "tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = str(tmp)
    import tempfile

    tempfile.tempdir = str(tmp)
    # Dataloader workers sample on CPU; keep each from spawning a thread per core.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    for path in (str(VILLA_DIR / "vesuvius" / "src"), str(SHIM_DIR), str(ROOT)):
        if path not in sys.path:
            sys.path.insert(0, path)


def pick_device(requested: str = "auto"):
    import torch

    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
