"""Shared CLI/runtime override helpers for simulation drivers."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any


def add_runtime_override_args(parser):
    parser.add_argument("--n-experiments", type=int, default=None, help="Override mc_opts.n_experiments.")
    parser.add_argument("--seed", type=int, default=None, help="Override mc_opts.seed.")
    parser.add_argument("--smoke-test", action="store_true", help="Run a moderate smoke preset.")
    parser.add_argument(
        "--force-rerun",
        action="store_true",
        help="Ignore cached .jbl results and recompute, overwriting cache files.",
    )

    # Explicit smoke knobs so callers can tune the preset.
    parser.add_argument(
        "--smoke-n-experiments",
        type=int,
        default=5,
        help="Smoke preset for mc_opts.n_experiments (default: 5).",
    )
    parser.add_argument(
        "--smoke-max-n-samples",
        type=int,
        default=500,
        help="Cap dgp_opts.n_samples at this value in smoke mode.",
    )
    parser.add_argument(
        "--smoke-max-n-test",
        type=int,
        default=500,
        help="Cap dgp_opts.n_test at this value in smoke mode.",
    )
    parser.add_argument(
        "--smoke-max-n-epochs",
        type=int,
        default=150,
        help="Cap method_opts n_epochs values at this value in smoke mode.",
    )
    parser.add_argument(
        "--smoke-max-burnin",
        type=int,
        default=100,
        help="Cap method_opts burnin values at this value in smoke mode.",
    )


def normalize_config_module(config_arg: str) -> str:
    """Normalize config input to an importable module path.

    Supports:
    - config_np_benchmark
    - config_np_benchmark.py
    - simulations/config_np_benchmark.py
    - simulations.config_np_benchmark
    """
    value = (config_arg or "").strip()
    if not value:
        raise ValueError("`--config` cannot be empty.")

    value = value.replace("\\", "/")
    if value.endswith(".py"):
        value = value[:-3]
    while value.startswith("./"):
        value = value[2:]

    if value.startswith("simulations."):
        return value
    if value.startswith("simulations/"):
        return value.replace("/", ".")
    if value.startswith("config_"):
        return f"simulations.{value}"

    if "/" in value:
        file_like = Path(value)
        if not file_like.is_absolute():
            file_like = (Path.cwd() / file_like).resolve()
        else:
            file_like = file_like.resolve()

        repo_root = Path(__file__).resolve().parent.parent
        if file_like.exists():
            try:
                rel = file_like.relative_to(repo_root)
                return ".".join(rel.with_suffix("").parts)
            except ValueError:
                pass

        module = value.replace("/", ".").lstrip(".")
        return module

    return value


def prepare_runtime_config(base_config: dict[str, Any], args) -> dict[str, Any]:
    """Return a copied config with optional runtime overrides applied."""
    config = deepcopy(base_config)
    config.setdefault("mc_opts", {})
    config.setdefault("dgp_opts", {})
    config.setdefault("method_opts", {})

    if args.smoke_test:
        _apply_smoke_overrides(config, args)
    if args.n_experiments is not None:
        config["mc_opts"]["n_experiments"] = int(args.n_experiments)
    if args.seed is not None:
        config["mc_opts"]["seed"] = int(args.seed)
    if getattr(args, "force_rerun", False):
        config["reload_results"] = False

    return config


def _cap_if_present(mapping: dict[str, Any], key: str, cap: int):
    if key not in mapping:
        return
    try:
        value = int(mapping[key])
    except Exception:
        return
    mapping[key] = min(value, cap)


def _to_int_if_present(mapping: Any, key: str):
    if not isinstance(mapping, dict) or key not in mapping:
        return None
    try:
        return int(mapping[key])
    except Exception:
        return None


def _walk_values(node: Any):
    if isinstance(node, dict):
        for value in node.values():
            yield from _walk_values(value)
    elif isinstance(node, (list, tuple)):
        for value in node:
            yield from _walk_values(value)
    else:
        yield node


def _apply_reg2sls_smoke_tuning(config: dict[str, Any], reg_cv: int, reg_n_alphas: int):
    methods = config.get("methods")
    if not isinstance(methods, dict):
        return

    for candidate in _walk_values(methods):
        cls = getattr(candidate, "__class__", None)
        if cls is None:
            continue
        if cls.__name__ != "regtsls":
            continue

        # Semiparametric Reg2SLS stores estimator instances in config methods.
        if hasattr(candidate, "cv"):
            candidate.cv = max(2, int(reg_cv))
        if hasattr(candidate, "n_alphas"):
            candidate.n_alphas = max(5, int(reg_n_alphas))


def _ensure_burnin_less_than_epochs(method_opts: dict[str, Any]):
    """Avoid empty averaging windows when burn-in equals/exceeds epoch count."""
    direct_epochs = _to_int_if_present(method_opts, "n_epochs")
    direct_burnin = _to_int_if_present(method_opts, "burnin")
    if direct_epochs is not None and direct_burnin is not None and direct_burnin >= direct_epochs:
        method_opts["burnin"] = max(0, direct_epochs - 1)

    fitargs = method_opts.get("fitargs")
    opts = method_opts.get("opts")
    fitargs_epochs = _to_int_if_present(fitargs, "n_epochs")
    opts_burnin = _to_int_if_present(opts, "burnin")
    if fitargs_epochs is not None and opts_burnin is not None and opts_burnin >= fitargs_epochs:
        opts["burnin"] = max(0, fitargs_epochs - 1)


def _apply_smoke_overrides(config: dict[str, Any], args):
    mc_opts = config.setdefault("mc_opts", {})
    dgp_opts = config.setdefault("dgp_opts", {})
    method_opts = config.setdefault("method_opts", {})

    mc_opts["n_experiments"] = int(args.smoke_n_experiments)
    _cap_if_present(dgp_opts, "n_samples", int(args.smoke_max_n_samples))
    _cap_if_present(dgp_opts, "n_test", int(args.smoke_max_n_test))
    _cap_if_present(method_opts, "n_epochs", int(args.smoke_max_n_epochs))
    _cap_if_present(method_opts, "burnin", int(args.smoke_max_burnin))

    fitargs = method_opts.get("fitargs")
    if isinstance(fitargs, dict):
        _cap_if_present(fitargs, "n_epochs", int(args.smoke_max_n_epochs))

    opts = method_opts.get("opts")
    if isinstance(opts, dict):
        _cap_if_present(opts, "burnin", int(args.smoke_max_burnin))

    # Reg2SLS smoke speedup: keep behavior but reduce CV workload.
    method_opts.setdefault("reg_cv", 2)
    method_opts.setdefault("reg_n_alphas", 50)
    _apply_reg2sls_smoke_tuning(
        config,
        reg_cv=int(method_opts["reg_cv"]),
        reg_n_alphas=int(method_opts["reg_n_alphas"]),
    )

    _ensure_burnin_less_than_epochs(method_opts)
