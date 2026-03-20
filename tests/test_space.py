"""Tests for lazyopt.space."""

import pytest

from lazyopt.space import (
    build_hebo_space,
    collect_search_space,
    load_yaml_config,
    parse_hp_calls,
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

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError, match="Source file not found"):
            parse_hp_calls("/nonexistent/file.py")

    def test_variable_ref_raises(self, tmp_path):
        code = """
from lazyopt import hp
DEFAULT = 0.1
lr = hp("lr", "float", DEFAULT)
"""
        p = tmp_path / "varref.py"
        p.write_text(code)
        with pytest.raises(ValueError, match="must be a literal"):
            parse_hp_calls(str(p))


class TestLoadYaml:
    def test_loads_params(self, sample_yaml):
        params = load_yaml_config(sample_yaml)
        assert "my_model.lr" in params
        assert "my_model.depth" in params
        assert params["my_model.lr"]["values"] == [0.01, 0.1, 1.0]

    def test_missing_yaml_raises(self):
        with pytest.raises(FileNotFoundError, match="YAML config not found"):
            load_yaml_config("/nonexistent.yaml")

    def test_empty_yaml_raises(self, tmp_path):
        p = tmp_path / "empty.yaml"
        p.write_text("")
        with pytest.raises(ValueError, match="must be a mapping"):
            load_yaml_config(str(p))

    def test_missing_keys_raises(self, tmp_path):
        content = """
ns:
  param:
    values: [1, 2, 3]
"""
        p = tmp_path / "bad.yaml"
        p.write_text(content)
        with pytest.raises(ValueError, match="missing required keys"):
            load_yaml_config(str(p))


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

    def test_empty_values_raises(self, tmp_path):
        code = """
from lazyopt import hp
lr = hp("lr", "float", 0.1, values=[])
"""
        p = tmp_path / "empty_grid.py"
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
        assert space.num_paras == 2

    def test_empty_values_raises(self):
        params = [{"qualified_name": "m.x", "values": []}]
        with pytest.raises(ValueError, match="empty values list"):
            build_hebo_space(params)
