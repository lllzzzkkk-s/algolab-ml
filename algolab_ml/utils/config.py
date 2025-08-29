from __future__ import annotations
from pathlib import Path
import json
from typing import Any, Dict

def load_json_or_yaml(path_or_text: str | dict | None) -> Dict[str, Any]:
    if path_or_text is None:
        return {}
    if isinstance(path_or_text, dict):
        return path_or_text
    s = str(path_or_text).strip()
    if not s:
        return {}
    if s.startswith("@"):
        p = Path(s[1:])
        if not p.exists():
            raise FileNotFoundError(f"Config file not found: {p}")
        if p.suffix.lower() in (".yaml", ".yml"):
            import yaml
            return dict(yaml.safe_load(p.read_text(encoding="utf-8")) or {})
        else:
            return dict(json.loads(p.read_text(encoding="utf-8")))
    try:
        return dict(json.loads(s))
    except Exception:
        return {}
