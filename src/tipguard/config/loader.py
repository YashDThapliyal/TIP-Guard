"""Load and validate YAML files into pydantic models with file-aware errors."""

from pathlib import Path

import yaml
from pydantic import BaseModel, ValidationError


class ConfigError(ValueError):
    """Raised when a configuration file is missing or invalid."""


def load_yaml_model[T: BaseModel](path: Path, model: type[T]) -> T:
    if not path.is_file():
        raise ConfigError(f"{path}: file not found")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path}: invalid YAML: {exc}") from exc
    if raw is None:
        raw = {}
    try:
        return model.model_validate(raw)
    except ValidationError as exc:
        details = "; ".join(
            f"{'.'.join(str(loc) for loc in err['loc'])}: {err['msg']}" for err in exc.errors()
        )
        raise ConfigError(f"{path}: {details}") from exc
