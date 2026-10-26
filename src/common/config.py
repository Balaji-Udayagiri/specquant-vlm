"""Load a YAML config into a plain dict."""
import pathlib
import yaml

def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def project_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[2]
