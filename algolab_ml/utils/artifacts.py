from __future__ import annotations
from pathlib import Path
import json, sys, platform, importlib, subprocess
import joblib
from datetime import datetime

def dump_json(path: Path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

def dump_joblib(path: Path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)

def versions_summary():
    pkgs = ["python", "pandas", "numpy", "scikit-learn", "xgboost", "lightgbm", "catboost"]
    out = {"platform": platform.platform()}
    for name in pkgs:
        try:
            if name == "python":
                out["python"] = sys.version
            else:
                m = importlib.import_module(name.replace("-", "_"))
                out[name] = getattr(m, "__version__", "unknown")
        except Exception:
            out[name] = "missing"
    return out

def git_commit_sha() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None

def make_run_dir(root: Path) -> Path:
    root = Path(root)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    d = root / ts
    d.mkdir(parents=True, exist_ok=True)
    return d
