"""读取 station.env（Python 侧），与 common.sh 同一规则：同名环境变量优先。"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def load(path: str | os.PathLike | None = None) -> dict[str, str]:
    path = Path(path or os.environ.get("STATION_ENV") or ROOT / "station.env")
    cfg: dict[str, str] = {}
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            key, val = key.strip(), val.strip()
            if len(val) >= 2 and val[0] == val[-1] and val[0] in "\"'":
                val = val[1:-1]
            cfg[key] = val
    for key in list(cfg):
        if key in os.environ:
            cfg[key] = os.environ[key]
    return cfg


CFG = load()


def get(key: str, default: str = "") -> str:
    return CFG.get(key) or os.environ.get(key) or default
