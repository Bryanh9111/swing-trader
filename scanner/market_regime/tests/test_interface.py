from __future__ import annotations

from scanner.market_regime.interface import RegimeConfig


def test_regime_config_to_mapping_roundtrip() -> None:
    cfg = RegimeConfig(
        regime_name="unit",
        description="unit",
        enabled=True,
        scanner={"box_threshold": 0.06},
        strategy={"min_score_threshold": 0.85},
        breakthrough={"require_confirmation_days": 1},
        risk_gate={"max_total_exposure": 0.85},
    )
    mapping = cfg.to_mapping()
    assert mapping["regime_name"] == "unit"
    assert mapping["scanner"]["box_threshold"] == 0.06
    assert mapping["risk_gate"]["max_total_exposure"] == 0.85
