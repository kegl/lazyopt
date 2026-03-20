"""Tests for lazyopt.space."""

import pytest

from lazyopt.space import (
    parse_hp_calls,
    load_yaml_config,
    collect_search_space,
    build_hebo_space,
)


@pytest.fixture
def sample_source(tmp_path):
    code = """
from lazyopt import hp

lr = hp("lr", "float", 0.1, values=[0.01, 0.1, 1.0])
depth = hp("depth", "int", 5, values=[3, 5, 7])
"""
    p = tmp_path / "my_model.py"
    p.write_text(code)
    return str(p)


@pytest.fixture
def sample_yaml(tmp_path):
    content = """
my_model:
  lr:
    dtype: float
    default: 0.1
    values: [0.01, 0.1, 1.0]
  depth:
    dtype: int
    default: 5
    values: [3, 5, 7]
"""
    p = tmp_path / "params.yaml"
    p.write_text(content)
    return str(p)


class TestParseHpCalls:
    def test_extracts_params(self, sample_source):
        params = parse_hp_calls(sample_source)
        assert len(params) == 2
        names = {p["name"] for p in params}
        assert names == {"lr", "depth"}

    def test_namespace_from_filename(self, sample_source):
        params = parse_hp_calls(sample_source)
        for p in params:
            assert p["namespace"] == "my_model"

    def test_values_extracted(self, sample_source):
        params = parse_hp_calls(sample_source)
        lr_param = next(p for p in params if p["name"] == "lr")
        assert lr_param["values"] == [0.01, 0.1, 1.0]
        assert lr_param["default"] == 0.1

    def test_keyword_args(self, tmp_path):
        code = """
from lazyopt import hp
x = hp(name="x", dtype="int", default=1, values=[1, 2, 3])
"""
        p = tmp_path / "kw_model.py"
        p.write_text(code)
        params = parse_hp_calls(str(p))
        assert len(params) == 1
        assert params[0]["name"] == "x"
        assert params[0]["values"] == [1, 2, 3]


class TestLoadYaml:
    def test_loads_params(self, sample_yaml):
        params = load_yaml_config(sample_yaml)
        assert "my_model.lr" in params
        assert "my_model.depth" in params
        assert params["my_model.lr"]["values"] == [0.01, 0.1, 1.0]


class TestCollectSearchSpace:
    def test_inline_values(self, sample_source):
        params = collect_search_space([sample_source])
        assert len(params) == 2

    def test_yaml_fallback(self, tmp_path, sample_yaml):
        code = """
from lazyopt import hp
lr = hp("lr", "float", 0.1)
"""
        p = tmp_path / "my_model.py"
        p.write_text(code)
        params = collect_search_space([str(p)], yaml_config=sample_yaml)
        lr = next(p for p in params if p["name"] == "lr")
        assert lr["values"] == [0.01, 0.1, 1.0]

    def test_missing_values_raises(self, tmp_path):
        code = """
from lazyopt import hp
lr = hp("lr", "float", 0.1)
"""
        p = tmp_path / "no_grid.py"
        p.write_text(code)
        with pytest.raises(ValueError, match="no grid values"):
            collect_search_space([str(p)])


class TestBuildHeboSpace:
    def test_builds_space(self):
        params = [
            {"qualified_name": "m.lr", "values": [0.01, 0.1, 1.0]},
            {"qualified_name": "m.depth", "values": [3, 5, 7]},
        ]
        space, idx_map = build_hebo_space(params)
        assert "m.lr" in idx_map
        assert idx_map["m.lr"] == [0.01, 0.1, 1.0]
        assert idx_map["m.depth"] == [3, 5, 7]
