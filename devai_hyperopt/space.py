"""AST-based hp() extraction, YAML loading, and HEBO DesignSpace builder."""

import ast
import yaml
from pathlib import Path
from hebo.design_space.design_space import DesignSpace


def parse_hp_calls(source_file: str) -> list[dict]:
    """AST-walk a Python file to extract all hp() calls statically.

    Returns a list of dicts with keys: name, namespace, dtype, default, values.
    """
    source_path = Path(source_file)
    namespace = source_path.stem
    tree = ast.parse(source_path.read_text())

    params = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name) and func.id == "hp":
            pass
        elif isinstance(func, ast.Attribute) and func.attr == "hp":
            pass
        else:
            continue

        kw = {k.arg: k.value for k in node.keywords}
        args = node.args

        name = _eval_literal(args[0] if len(args) > 0 else kw.get("name"))
        dtype = _eval_literal(args[1] if len(args) > 1 else kw.get("dtype"))
        default = _eval_literal(args[2] if len(args) > 2 else kw.get("default"))
        values_node = args[3] if len(args) > 3 else kw.get("values")
        values = _eval_literal(values_node) if values_node is not None else None

        params.append(
            {
                "name": name,
                "namespace": namespace,
                "qualified_name": f"{namespace}.{name}",
                "dtype": dtype,
                "default": default,
                "values": values,
            }
        )
    return params


def _eval_literal(node):
    """Safely evaluate an AST node to a Python literal."""
    if node is None:
        return None
    return ast.literal_eval(node)


def load_yaml_config(yaml_path: str) -> dict[str, dict]:
    """Load a YAML grid config file.

    Expected format:
        namespace:
          param_name:
            dtype: float
            default: 0.1
            values: [0.01, 0.05, 0.1, 0.5]

    Returns a dict mapping qualified_name -> param dict.
    """
    with open(yaml_path) as f:
        raw = yaml.safe_load(f)

    params = {}
    for namespace, param_defs in raw.items():
        for name, spec in param_defs.items():
            qname = f"{namespace}.{name}"
            params[qname] = {
                "name": name,
                "namespace": namespace,
                "qualified_name": qname,
                "dtype": spec["dtype"],
                "default": spec["default"],
                "values": spec.get("values"),
            }
    return params


def collect_search_space(
    source_files: list[str],
    yaml_config: str | None = None,
) -> list[dict]:
    """Merge inline hp() grids with optional YAML fallback.

    Inline values take precedence. Raises ValueError if a param
    has no grid (values is None) anywhere.
    """
    yaml_params = {}
    if yaml_config is not None:
        yaml_params = load_yaml_config(yaml_config)

    all_params: dict[str, dict] = {}
    for src in source_files:
        for p in parse_hp_calls(src):
            all_params[p["qualified_name"]] = p

    for qname, p in all_params.items():
        if p["values"] is None and qname in yaml_params:
            p["values"] = yaml_params[qname]["values"]

    for qname, p in all_params.items():
        if p["values"] is None or len(p["values"]) == 0:
            raise ValueError(
                f"Hyperparameter {qname!r} has no grid values. "
                "Provide values inline or in a YAML config."
            )

    return list(all_params.values())


def build_hebo_space(
    params: list[dict],
) -> tuple[DesignSpace, dict[str, list]]:
    """Convert parameter dicts to a HEBO DesignSpace using int indices.

    Returns (design_space, index_to_value_map) where
    index_to_value_map maps qualified_name -> list of actual values.
    """
    space_cfg = []
    index_to_value: dict[str, list] = {}

    for p in params:
        qname = p["qualified_name"]
        vals = list(p["values"])
        index_to_value[qname] = vals
        space_cfg.append(
            {
                "name": qname,
                "type": "int",
                "lb": 0,
                "ub": len(vals) - 1,
            }
        )

    space = DesignSpace().parse(space_cfg)
    return space, index_to_value
