"""Regime configuration loader for `config/regimes/*.yaml`."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from common.interface import Result, ResultStatus

from .interface import MarketRegime, RegimeConfig

__all__ = ["RegimeConfigLoader"]


def _ensure_mapping(value: Any, *, source: str) -> Result[dict[str, Any]]:
    if value is None:
        return Result.success({})
    if isinstance(value, dict):
        return Result.success(dict(value))
    return Result.failed(TypeError(f"Expected mapping in {source}, got {type(value)!r}"), "INVALID_YAML_MAPPING")


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


@dataclass(frozen=True, slots=True)
class RegimeConfigLoader:
    """Load and merge market regime YAML configurations."""

    base_dir: Path = Path("config/regimes")
    base_filename: str = "base.yaml"

    def list_regime_names(self) -> list[str]:
        """List available regime names based on YAML files in `base_dir`."""

        if not self.base_dir.exists():
            return []
        candidates = []
        for path in sorted(self.base_dir.glob("*.yaml")):
            if path.name == self.base_filename:
                continue
            candidates.append(path.stem)
        return candidates

    def load(self, regime: MarketRegime | str) -> Result[RegimeConfig]:
        """Load the merged configuration for `regime`."""

        regime_name = self._normalize_regime_name(regime)
        base_path = self.base_dir / self.base_filename
        overlay_path = self.base_dir / f"{regime_name}.yaml"

        base_loaded = self._load_yaml_mapping(base_path)
        if base_loaded.status is ResultStatus.FAILED:
            return Result.failed(base_loaded.error or ValueError("base config load failed"), base_loaded.reason_code or "BASE_LOAD_FAILED")

        if regime_name == "base":
            merged = base_loaded.data
            return self._to_regime_config(merged, fallback_name="base")

        overlay_loaded = self._load_yaml_mapping(overlay_path)
        if overlay_loaded.status is ResultStatus.FAILED:
            return Result.failed(
                overlay_loaded.error or ValueError("overlay config load failed"),
                overlay_loaded.reason_code or "OVERLAY_LOAD_FAILED",
            )

        overlay = overlay_loaded.data
        inherited_name = overlay.get("inherits", "base")
        inherited = base_loaded.data
        if isinstance(inherited_name, str) and inherited_name not in ("", "base"):
            inherited_path = self.base_dir / f"{inherited_name}.yaml"
            inherited_loaded = self._load_yaml_mapping(inherited_path)
            if inherited_loaded.status is ResultStatus.FAILED:
                return Result.failed(
                    inherited_loaded.error or ValueError("inherited config load failed"),
                    inherited_loaded.reason_code or "INHERITED_LOAD_FAILED",
                )
            inherited = _deep_merge(inherited, inherited_loaded.data)

        merged = _deep_merge(inherited, overlay)
        merged.pop("inherits", None)
        return self._to_regime_config(merged, fallback_name=regime_name)

    def _load_yaml_mapping(self, path: Path) -> Result[dict[str, Any]]:
        if not path.exists():
            return Result.failed(FileNotFoundError(str(path)), "CONFIG_NOT_FOUND")
        try:
            with path.open("r", encoding="utf-8") as file:
                loaded = yaml.safe_load(file)
        except Exception as exc:  # noqa: BLE001
            return Result.failed(exc, "CONFIG_READ_FAILED")

        mapping_result = _ensure_mapping(loaded, source=str(path))
        if mapping_result.status is ResultStatus.FAILED:
            return mapping_result
        return Result.success(mapping_result.data)

    @staticmethod
    def _normalize_regime_name(regime: MarketRegime | str) -> str:
        if isinstance(regime, MarketRegime):
            if regime is MarketRegime.BULL:
                return "bull_market"
            if regime is MarketRegime.BEAR:
                return "bear_market"
            if regime is MarketRegime.CHOPPY:
                return "choppy_market"
            return "base"
        return str(regime).strip()

    @staticmethod
    def _to_regime_config(mapping: Mapping[str, Any], *, fallback_name: str) -> Result[RegimeConfig]:
        regime_name = str(mapping.get("regime_name") or fallback_name)
        description = str(mapping.get("description") or "")
        enabled_raw = mapping.get("enabled", True)
        enabled = bool(enabled_raw) if isinstance(enabled_raw, (bool, int)) else True

        scanner = mapping.get("scanner", {})
        market_regime = mapping.get("market_regime", {})
        strategy = mapping.get("strategy", {})
        breakthrough = mapping.get("breakthrough", {})
        risk_gate = mapping.get("risk_gate", {})
        # V21: BULL confirmation and risk overlay
        bull_confirmation = mapping.get("bull_confirmation", {})
        risk_overlay = mapping.get("risk_overlay", {})
        # V21: Regime-aware capital allocation overrides
        capital_allocation = mapping.get("capital_allocation", {})

        if scanner is None:
            scanner = {}
        if market_regime is None:
            market_regime = {}
        if strategy is None:
            strategy = {}
        if breakthrough is None:
            breakthrough = {}
        if risk_gate is None:
            risk_gate = {}
        if bull_confirmation is None:
            bull_confirmation = {}
        if risk_overlay is None:
            risk_overlay = {}
        if capital_allocation is None:
            capital_allocation = {}

        if (
            not isinstance(scanner, dict)
            or not isinstance(market_regime, dict)
            or not isinstance(strategy, dict)
            or not isinstance(breakthrough, dict)
            or not isinstance(risk_gate, dict)
            or not isinstance(bull_confirmation, dict)
            or not isinstance(risk_overlay, dict)
            or not isinstance(capital_allocation, dict)
        ):
            return Result.failed(
                ValueError("scanner/market_regime/strategy/breakthrough/risk_gate/bull_confirmation/risk_overlay/capital_allocation must be mappings"),
                "INVALID_SECTION_TYPE",
            )

        return Result.success(
            RegimeConfig(
                regime_name=regime_name,
                description=description,
                enabled=enabled,
                scanner=dict(scanner),
                market_regime=dict(market_regime),
                strategy=dict(strategy),
                breakthrough=dict(breakthrough),
                risk_gate=dict(risk_gate),
                bull_confirmation=dict(bull_confirmation),
                risk_overlay=dict(risk_overlay),
                capital_allocation=dict(capital_allocation),
            )
        )
