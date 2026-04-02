from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import msgspec
import pytest
import yaml

from common.config import (
    BaseConfig,
    ConfigFileNotFoundError,
    ConfigLoaderError,
    ConfigValidationError,
    YAMLConfigLoader,
)
from common.interface import ResultStatus


def _write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def _create_config_dir(
    base_dir: Path,
    base_data: dict[str, Any],
    *,
    env_name: str = "dev",
    env_override: dict[str, Any] | None = None,
    local_override: dict[str, Any] | None = None,
) -> Path:
    config_dir = base_dir / "config"
    config_dir.mkdir(parents=True, exist_ok=True)

    base_contents = copy.deepcopy(base_data)
    strategy_config = base_contents.get("strategy")
    if isinstance(strategy_config, dict):
        entry_config = strategy_config.get("entry")
        if isinstance(entry_config, dict):
            max_position = entry_config.get("max_position_size")
            first_tranche = entry_config.get("first_tranche_pct")
            if (
                isinstance(max_position, (int, float))
                and isinstance(first_tranche, (int, float))
                and first_tranche > max_position
            ):
                entry_config["first_tranche_pct"] = max_position

    _write_yaml(config_dir / "config.yaml", base_contents)

    if env_override is not None:
        _write_yaml(config_dir / f"config.{env_name}.yaml", env_override)
    else:
        # Create empty env-specific override file to satisfy loader expectations.
        _write_yaml(config_dir / f"config.{env_name}.yaml", {})

    if local_override is not None:
        _write_yaml(config_dir / "config.local.yaml", local_override)

    return config_dir


@pytest.fixture(scope="module")
def base_config_data() -> dict[str, Any]:
    project_root = Path(__file__).resolve().parents[2]
    base_path = project_root / "config" / "config.yaml"
    return yaml.safe_load(base_path.read_text(encoding="utf-8"))


def test_load_config_layers_env_overrides(tmp_path: Path, base_config_data: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    base = copy.deepcopy(base_config_data)

    env_override = {
        "system": {"log_level": "WARNING"},
        "data_sources": {"failover": {"max_retries": 5}},
        "scanner": {"lookback_days": 120},
    }
    local_override = {
        "risk_gate": {"portfolio": {"max_total_exposure": 0.85}},
        "journal": None,
    }

    config_dir = _create_config_dir(
        tmp_path,
        base,
        env_override=env_override,
        local_override=local_override,
    )

    monkeypatch.setenv("AST__SYSTEM__LOG_LEVEL", "ERROR")
    monkeypatch.setenv("AST__DEV__SCANNER__ENABLED", "false")
    monkeypatch.setenv("AST__DATA_SOURCES__FAILOVER__RETRY_DELAY_SECONDS", "10")

    loader = YAMLConfigLoader(config_dir=config_dir)

    result = loader.load_config_result("dev")
    assert result.status is ResultStatus.SUCCESS
    assert isinstance(result.data, BaseConfig)
    assert result.error is None

    config = result.data
    assert config.system.log_level == "ERROR"
    assert config.scanner.enabled is False
    assert config.scanner.lookback_days == 120
    assert config.data_sources.failover.max_retries == 5
    assert config.data_sources.failover.retry_delay_seconds == 10
    assert config.risk_gate.portfolio.max_total_exposure == pytest.approx(0.85)
    assert config.journal.enabled is base_config_data["journal"]["enabled"]


def test_env_specific_overrides_take_priority(tmp_path: Path, base_config_data: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    base = copy.deepcopy(base_config_data)
    base["scanner"]["enabled"] = True

    config_dir = _create_config_dir(tmp_path, base)

    monkeypatch.setenv("AST__SCANNER__ENABLED", "true")
    monkeypatch.setenv("AST__DEV__SCANNER__ENABLED", "false")

    loader = YAMLConfigLoader(config_dir=config_dir)
    config = loader.load_config("dev")

    assert config.scanner.enabled is False


def test_load_config_validation_error(tmp_path: Path, base_config_data: dict[str, Any]) -> None:
    base = copy.deepcopy(base_config_data)
    base["strategy"]["entry"]["max_position_size"] = 2.0

    config_dir = _create_config_dir(tmp_path, base)
    loader = YAMLConfigLoader(config_dir=config_dir)

    result = loader.load_config_result("dev")

    assert result.status is ResultStatus.FAILED
    assert isinstance(result.error, ConfigValidationError)
    assert result.reason_code == "CONFIG_LOAD_ERROR"


def test_load_config_msgspec_validation_error(tmp_path: Path, base_config_data: dict[str, Any]) -> None:
    base = copy.deepcopy(base_config_data)
    base["scanner"]["enabled"] = "yes"

    config_dir = _create_config_dir(tmp_path, base)
    loader = YAMLConfigLoader(config_dir=config_dir)

    result = loader.load_config_result("dev")

    assert result.status is ResultStatus.FAILED
    assert isinstance(result.error, msgspec.ValidationError)
    assert result.reason_code == "CONFIG_VALIDATION_ERROR"


def test_load_config_file_not_found(tmp_path: Path) -> None:
    config_dir = tmp_path / "missing_config"
    config_dir.mkdir()

    loader = YAMLConfigLoader(config_dir=config_dir)

    with pytest.raises(ConfigFileNotFoundError):
        loader.load_config("dev")


def test_loader_directory_absent(tmp_path: Path) -> None:
    with pytest.raises(ConfigLoaderError):
        YAMLConfigLoader(config_dir=tmp_path / "does_not_exist")


def test_load_yaml_requires_mapping(tmp_path: Path) -> None:
    config_dir = tmp_path / "invalid_yaml"
    config_dir.mkdir()
    (config_dir / "config.yaml").write_text("- just-a-list\n- not-a-mapping\n", encoding="utf-8")

    loader = YAMLConfigLoader(config_dir=config_dir)

    with pytest.raises(ConfigLoaderError):
        loader.load_config("dev")


def test_load_config_result_unexpected_error(tmp_path: Path, base_config_data: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    base = copy.deepcopy(base_config_data)
    config_dir = _create_config_dir(tmp_path, base)

    loader = YAMLConfigLoader(config_dir=config_dir)

    def _boom(_: str) -> dict[str, Any]:
        raise RuntimeError("boom")

    monkeypatch.setattr(loader, "_load_and_merge_layers", _boom)

    result = loader.load_config_result("dev")
    assert result.status is ResultStatus.FAILED
    assert result.reason_code == "CONFIG_UNEXPECTED_ERROR"
    assert isinstance(result.error, RuntimeError)


def test_deep_merge_skips_none_overrides(tmp_path: Path, base_config_data: dict[str, Any]) -> None:
    base = copy.deepcopy(base_config_data)
    local_override = {"journal": None}

    config_dir = _create_config_dir(tmp_path, base, local_override=local_override)
    loader = YAMLConfigLoader(config_dir=config_dir)

    config = loader.load_config("dev")
    assert config.journal.storage_path == base_config_data["journal"]["storage_path"]


def test_dotenv_loading(tmp_path: Path, base_config_data: dict[str, Any]) -> None:
    base = copy.deepcopy(base_config_data)
    env_key = "AST_TEST_DOTENV_SECRET"

    custom_dotenv = tmp_path / "custom.env"
    custom_dotenv.write_text(f"{env_key}=custom\n", encoding="utf-8")

    config_dir = _create_config_dir(tmp_path, base)
    (config_dir / "secrets.env").write_text(f"{env_key}=from_secrets\n", encoding="utf-8")

    try:
        loader = YAMLConfigLoader(config_dir=config_dir, dotenv_path=custom_dotenv)
        assert os.environ[env_key] == "custom"
        loader.load_config("dev")  # ensure overall load still succeeds
    finally:
        os.environ.pop(env_key, None)


def test_load_config_result_returns_data(tmp_path: Path, base_config_data: dict[str, Any]) -> None:
    base = copy.deepcopy(base_config_data)
    config_dir = _create_config_dir(tmp_path, base)

    loader = YAMLConfigLoader(config_dir=config_dir)

    result = loader.load_config_result("dev")
    assert result.status is ResultStatus.SUCCESS
    assert isinstance(result.data, BaseConfig)
    assert result.error is None
