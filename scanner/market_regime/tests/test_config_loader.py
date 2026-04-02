from __future__ import annotations

from pathlib import Path

from common.interface import ResultStatus
from scanner.market_regime.interface import MarketRegime
from scanner.market_regime.config_loader import RegimeConfigLoader, _deep_merge, _ensure_mapping


def test__deep_merge_merges_nested_mappings() -> None:
    base = {"a": {"x": 1, "y": 2}, "b": 1}
    override = {"a": {"y": 99}, "c": 3}
    merged = _deep_merge(base, override)
    assert merged == {"a": {"x": 1, "y": 99}, "b": 1, "c": 3}


def test__ensure_mapping_accepts_none_as_empty() -> None:
    result = _ensure_mapping(None, source="unit")
    assert result.status is ResultStatus.SUCCESS
    assert result.data == {}


def test__ensure_mapping_rejects_non_mapping() -> None:
    result = _ensure_mapping(["not", "a", "dict"], source="unit")
    assert result.status is ResultStatus.FAILED
    assert result.reason_code == "INVALID_YAML_MAPPING"


def test_loader_lists_repo_regime_configs() -> None:
    loader = RegimeConfigLoader(base_dir=Path("config/regimes"))
    names = loader.list_regime_names()
    assert "bull_market" in names
    assert "bear_market" in names
    assert "choppy_market" in names


def test_loader_lists_empty_when_base_dir_missing(tmp_path: Path) -> None:
    loader = RegimeConfigLoader(base_dir=tmp_path / "missing")
    assert loader.list_regime_names() == []


def test_loader_merges_base_and_override(tmp_path: Path) -> None:
    (tmp_path / "base.yaml").write_text(
        """
regime_name: "base"
description: "base"
scanner:
  box_threshold: 0.06
  ma_diff_threshold: 0.02
strategy:
  min_score_threshold: 0.85
breakthrough:
  require_confirmation_days: 1
""".lstrip(),
        encoding="utf-8",
    )
    (tmp_path / "bull_market.yaml").write_text(
        """
regime_name: "bull_market"
description: "override"
inherits: "base"
scanner:
  box_threshold: 0.08
strategy:
  min_score_threshold: 0.80
""".lstrip(),
        encoding="utf-8",
    )

    loader = RegimeConfigLoader(base_dir=tmp_path)
    result = loader.load("bull_market")
    assert result.status is ResultStatus.SUCCESS
    cfg = result.data
    assert cfg is not None
    assert cfg.regime_name == "bull_market"
    assert cfg.scanner["box_threshold"] == 0.08
    assert cfg.scanner["ma_diff_threshold"] == 0.02
    assert cfg.strategy["min_score_threshold"] == 0.80
    assert cfg.breakthrough["require_confirmation_days"] == 1


def test_loader_supports_two_level_inheritance(tmp_path: Path) -> None:
    (tmp_path / "base.yaml").write_text(
        """
regime_name: "base"
description: "base"
scanner:
  box_threshold: 0.06
  min_dollar_volume: 1000000
""".lstrip(),
        encoding="utf-8",
    )
    (tmp_path / "mid.yaml").write_text(
        """
regime_name: "mid"
description: "mid"
inherits: "base"
scanner:
  min_dollar_volume: 2000000
""".lstrip(),
        encoding="utf-8",
    )
    (tmp_path / "child.yaml").write_text(
        """
regime_name: "child"
description: "child"
inherits: "mid"
scanner:
  box_threshold: 0.04
""".lstrip(),
        encoding="utf-8",
    )

    loader = RegimeConfigLoader(base_dir=tmp_path)
    result = loader.load("child")
    assert result.status is ResultStatus.SUCCESS
    cfg = result.data
    assert cfg is not None
    assert cfg.scanner["box_threshold"] == 0.04
    assert cfg.scanner["min_dollar_volume"] == 2000000


def test_loader_loads_base_when_requested(tmp_path: Path) -> None:
    (tmp_path / "base.yaml").write_text(
        """
description: "base"
scanner:
  box_threshold: 0.06
""".lstrip(),
        encoding="utf-8",
    )
    loader = RegimeConfigLoader(base_dir=tmp_path)
    result = loader.load("base")
    assert result.status is ResultStatus.SUCCESS
    assert result.data is not None
    assert result.data.regime_name == "base"
    assert result.data.scanner["box_threshold"] == 0.06


def test_loader_returns_failed_when_overlay_missing(tmp_path: Path) -> None:
    (tmp_path / "base.yaml").write_text(
        """
regime_name: "base"
description: "base"
""".lstrip(),
        encoding="utf-8",
    )
    loader = RegimeConfigLoader(base_dir=tmp_path)
    result = loader.load("does_not_exist")
    assert result.status is ResultStatus.FAILED
    assert result.reason_code == "CONFIG_NOT_FOUND"


def test_loader_accepts_enum_input(tmp_path: Path) -> None:
    (tmp_path / "base.yaml").write_text(
        """
regime_name: "base"
description: "base"
scanner:
  box_threshold: 0.06
""".lstrip(),
        encoding="utf-8",
    )
    (tmp_path / "bull_market.yaml").write_text(
        """
regime_name: "bull_market"
description: "override"
inherits: "base"
scanner:
  box_threshold: 0.08
""".lstrip(),
        encoding="utf-8",
    )
    loader = RegimeConfigLoader(base_dir=tmp_path)
    result = loader.load(MarketRegime.BULL)
    assert result.status is ResultStatus.SUCCESS
    assert result.data is not None
    assert result.data.scanner["box_threshold"] == 0.08


def test_loader_maps_enum_variants(tmp_path: Path) -> None:
    (tmp_path / "base.yaml").write_text('regime_name: "base"\ndescription: "base"\n', encoding="utf-8")
    (tmp_path / "bear_market.yaml").write_text('regime_name: "bear_market"\ndescription: "bear"\n', encoding="utf-8")
    (tmp_path / "choppy_market.yaml").write_text('regime_name: "choppy_market"\ndescription: "chop"\n', encoding="utf-8")
    loader = RegimeConfigLoader(base_dir=tmp_path)

    bear = loader.load(MarketRegime.BEAR)
    assert bear.status is ResultStatus.SUCCESS
    assert bear.data is not None
    assert bear.data.regime_name == "bear_market"

    chop = loader.load(MarketRegime.CHOPPY)
    assert chop.status is ResultStatus.SUCCESS
    assert chop.data is not None
    assert chop.data.regime_name == "choppy_market"

    unknown = loader.load(MarketRegime.UNKNOWN)
    assert unknown.status is ResultStatus.SUCCESS
    assert unknown.data is not None
    assert unknown.data.regime_name == "base"


def test_loader_returns_read_failed_for_invalid_yaml(tmp_path: Path) -> None:
    (tmp_path / "base.yaml").write_text("scanner: [", encoding="utf-8")
    loader = RegimeConfigLoader(base_dir=tmp_path)
    result = loader.load("base")
    assert result.status is ResultStatus.FAILED
    assert result.reason_code == "CONFIG_READ_FAILED"


def test_loader_returns_invalid_mapping_when_yaml_is_sequence(tmp_path: Path) -> None:
    (tmp_path / "base.yaml").write_text("- a\n- b\n", encoding="utf-8")
    loader = RegimeConfigLoader(base_dir=tmp_path)
    result = loader.load("base")
    assert result.status is ResultStatus.FAILED
    assert result.reason_code == "INVALID_YAML_MAPPING"


def test_loader_returns_failed_when_inherited_missing(tmp_path: Path) -> None:
    (tmp_path / "base.yaml").write_text('regime_name: "base"\ndescription: "base"\n', encoding="utf-8")
    (tmp_path / "child.yaml").write_text(
        """
regime_name: "child"
description: "child"
inherits: "missing"
""".lstrip(),
        encoding="utf-8",
    )
    loader = RegimeConfigLoader(base_dir=tmp_path)
    result = loader.load("child")
    assert result.status is ResultStatus.FAILED
    assert result.reason_code == "CONFIG_NOT_FOUND"


def test_loader_rejects_invalid_section_types(tmp_path: Path) -> None:
    (tmp_path / "base.yaml").write_text('regime_name: "base"\ndescription: "base"\n', encoding="utf-8")
    (tmp_path / "bad.yaml").write_text(
        """
regime_name: "bad"
description: "bad"
scanner: []
""".lstrip(),
        encoding="utf-8",
    )
    loader = RegimeConfigLoader(base_dir=tmp_path)
    result = loader.load("bad")
    assert result.status is ResultStatus.FAILED
    assert result.reason_code == "INVALID_SECTION_TYPE"
