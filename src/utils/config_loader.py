from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml

_BASE_CONFIG = Path(__file__).parent.parent.parent / "config" / "base.yaml"


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base. Override wins on conflicts."""
    result = dict(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = val
    return result


def _to_namespace(d: Any) -> Any:
    """Recursively convert nested dicts to SimpleNamespace."""
    if isinstance(d, dict):
        return SimpleNamespace(**{k: _to_namespace(v) for k, v in d.items()})
    if isinstance(d, list):
        return [_to_namespace(v) for v in d]
    return d


def load_config(experiment_path: str | Path | None = None) -> SimpleNamespace:
    """Load and merge base config with an optional experiment-specific override.

    Args:
        experiment_path: Path to experiment YAML (relative or absolute).
                         If None, only base.yaml is loaded.

    Returns:
        Merged config as a SimpleNamespace (nested).
    """
    with open(_BASE_CONFIG, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if experiment_path is not None:
        exp_path = Path(experiment_path)
        if not exp_path.is_absolute():
            exp_path = Path(__file__).parent.parent.parent / exp_path
        with open(exp_path, encoding="utf-8") as f:
            exp_cfg = yaml.safe_load(f)
        cfg = _deep_merge(cfg, exp_cfg)

    return _to_namespace(cfg)
