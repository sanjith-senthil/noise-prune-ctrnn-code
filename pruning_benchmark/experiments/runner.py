"""Experiment runners for evaluating pruning strategies on CTRNNs."""

from __future__ import annotations

import json
import os
import random
import csv
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from ..analysis import compile_run_metrics, save_metrics, snapshot_model
from ..config import ExperimentConfig
from ..tasks import ModCogTrialDM
from ..tasks.modcog import resolve_modcog_callable
from ..models import CTRNN
from ..pruning import (
    PRUNE_AMOUNT_STEP,
    PruneContext,
    enforce_constraints,
    finalize_pruning,
    get_pruner,
    validate_prune_fraction,
)
from ..training import evaluate, last_valid_logits_targets, train_epoch
from ..utils import make_run_id, set_global_seed


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def _ensure_run_directory(run_id: str) -> Path:
    path = Path("results") / run_id
    path.mkdir(parents=True, exist_ok=True)
    (path / "plots").mkdir(exist_ok=True)
    return path


def _dataclass_to_dict(obj: Any) -> Dict[str, Any]:
    if is_dataclass(obj):
        return asdict(obj)
    return {}


def _format_yaml_scalar(value: Any) -> str:
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, (int, float)):
        return json.dumps(value)
    return json.dumps(value, sort_keys=True)


def _yaml_lines(data: Any, indent: int = 0) -> List[str]:
    prefix = "  " * indent
    if isinstance(data, dict):
        lines: List[str] = []
        for key in sorted(data.keys()):
            value = data[key]
            if isinstance(value, (dict, list)):
                lines.append(f"{prefix}{key}:")
                lines.extend(_yaml_lines(value, indent + 1))
            else:
                lines.append(f"{prefix}{key}: {_format_yaml_scalar(value)}")
        return lines
    if isinstance(data, list):
        lines: List[str] = []
        for item in data:
            if isinstance(item, (dict, list)):
                lines.append(f"{prefix}-")
                lines.extend(_yaml_lines(item, indent + 1))
            else:
                lines.append(f"{prefix}- {_format_yaml_scalar(item)}")
        return lines
    return [f"{prefix}{_format_yaml_scalar(data)}"]


def _write_config_snapshot(run_dir: Path, snapshot: Dict[str, Any]) -> Dict[str, Path]:
    json_path = run_dir / "config.json"
    json_path.write_text(json.dumps(snapshot, indent=2, sort_keys=True))
    yaml_path = run_dir / "config.yaml"
    yaml_content = "\n".join(["---"] + _yaml_lines(snapshot)) + "\n"
    yaml_path.write_text(yaml_content)
    return {"json": json_path, "yaml": yaml_path}


def _clone_state_dict_cpu(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def _metric_better(
    candidate: Dict[str, Any],
    incumbent: Dict[str, Any] | None,
    *,
    metric: str,
    mode: str,
    tie_metric: str | None = None,
    tie_mode: str = "min",
) -> bool:
    if incumbent is None:
        return True
    cand_value = candidate.get(metric)
    best_value = incumbent.get(metric)
    if cand_value is None:
        return False
    if best_value is None:
        return True
    cand_float = float(cand_value)
    best_float = float(best_value)
    if mode == "max":
        if cand_float > best_float:
            return True
        if cand_float < best_float:
            return False
    elif mode == "min":
        if cand_float < best_float:
            return True
        if cand_float > best_float:
            return False
    else:
        raise ValueError("best checkpoint metric mode must be 'max' or 'min'")
    if not tie_metric:
        return False
    cand_tie = candidate.get(tie_metric)
    best_tie = incumbent.get(tie_metric)
    if cand_tie is None:
        return False
    if best_tie is None:
        return True
    cand_tie_float = float(cand_tie)
    best_tie_float = float(best_tie)
    if tie_mode == "max":
        return cand_tie_float > best_tie_float
    if tie_mode == "min":
        return cand_tie_float < best_tie_float
    raise ValueError("best checkpoint tie metric mode must be 'max' or 'min'")


def _write_training_history(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in keys})


def _extract_prune_kwargs(strategy: str, options: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    prune_kwargs: Dict[str, Any] = {}
    prune_meta: Dict[str, Any] = {}
    if strategy in {
        "noise_prune",
        "noise_prune_capped_rescale",
        "vanilla_mask_only",
        "simulation_noise_prune_mask_only",
        "simulation_noise_prune_rescale",
        "simulation_noise_prune_capped_rescale",
    }:
        sigma = float(options.pop("noise_sigma", 1.0))
        eps = float(options.pop("noise_eps", 0.3))
        leak_shift = float(options.pop("noise_leak_shift", 0.0))
        matched_diagonal = bool(options.pop("noise_matched_diagonal", True))
        rng_seed = options.pop("noise_rng_seed", None)
        prune_kwargs.update({
            "sigma": sigma,
            "eps": eps,
            "leak_shift": leak_shift,
            "matched_diagonal": matched_diagonal,
        })
        prune_meta.update(prune_kwargs)
        if rng_seed is not None:
            rng_seed = int(rng_seed)
            prune_meta["rng_seed"] = rng_seed
            prune_kwargs["rng"] = np.random.default_rng(rng_seed)
        if strategy in {
            "simulation_noise_prune_mask_only",
            "simulation_noise_prune_rescale",
            "simulation_noise_prune_capped_rescale",
        }:
            sigma_source = str(options.pop("sim_np_sigma_source", "natural_voltage"))
            observable_space = str(options.pop("sim_np_observable_space", "rate"))
            inject_space = str(options.pop("sim_np_inject_space", "rate"))
            centering = str(options.pop("sim_np_centering", "trajectory_mean"))
            max_samples = int(options.pop("sim_np_max_samples", 25_000))
            burn_in_steps = int(options.pop("sim_np_burn_in_steps", 300))
            manual_sigma = options.pop("sim_np_sigma", None)
            # Simulation-based noise-prune defaults to empirical sigma matching;
            # only use the shared noise_sigma if the caller explicitly supplied sim_np_sigma.
            prune_kwargs.pop("sigma", None)
            prune_meta.pop("sigma", None)
            if manual_sigma is not None:
                prune_kwargs["sigma"] = float(manual_sigma)
                prune_meta["sim_np_sigma"] = float(manual_sigma)
            else:
                prune_meta["sim_np_sigma"] = None
            prune_kwargs.update({
                "sigma_source": sigma_source,
                "observable_space": observable_space,
                "inject_space": inject_space,
                "centering": centering,
                "max_samples": max_samples,
                "burn_in_steps": burn_in_steps,
            })
            if rng_seed is not None:
                prune_kwargs["rng_seed"] = rng_seed
            prune_meta.update({
                "sim_np_sigma_source": sigma_source,
                "sim_np_observable_space": observable_space,
                "sim_np_inject_space": inject_space,
                "sim_np_centering": centering,
                "sim_np_max_samples": max_samples,
                "sim_np_burn_in_steps": burn_in_steps,
            })
        else:
            for key in (
                "sim_np_sigma",
                "sim_np_sigma_source",
                "sim_np_observable_space",
                "sim_np_inject_space",
                "sim_np_centering",
                "sim_np_max_samples",
                "sim_np_burn_in_steps",
            ):
                options.pop(key, None)
    else:
        for key in (
            "noise_sigma",
            "noise_eps",
            "noise_leak_shift",
            "noise_matched_diagonal",
            "noise_rng_seed",
            "sim_np_sigma",
            "sim_np_sigma_source",
            "sim_np_observable_space",
            "sim_np_inject_space",
            "sim_np_centering",
            "sim_np_max_samples",
            "sim_np_burn_in_steps",
        ):
            options.pop(key, None)
    if strategy in {"noise_prune_capped_rescale", "simulation_noise_prune_capped_rescale"}:
        cap_mode = str(options.pop("rescale_cap_mode", "quantile"))
        cap_value_raw = options.pop("rescale_cap_value", None)
        cap_quantile_raw = options.pop("rescale_cap_quantile", 0.95)
        if cap_mode == "fixed":
            if cap_value_raw is None:
                raise ValueError("rescale_cap_value is required when rescale_cap_mode='fixed'.")
            cap_value = float(cap_value_raw)
            prune_kwargs["rescale_cap"] = cap_value
            prune_meta["rescale_cap_value"] = cap_value
            prune_meta["rescale_cap_quantile"] = None
        elif cap_mode == "quantile":
            cap_quantile = float(cap_quantile_raw)
            prune_kwargs["rescale_cap_quantile"] = cap_quantile
            prune_meta["rescale_cap_value"] = None
            prune_meta["rescale_cap_quantile"] = cap_quantile
        else:
            raise ValueError("rescale_cap_mode must be 'fixed' or 'quantile'.")
        prune_meta["rescale_cap_mode"] = cap_mode
    else:
        for key in ("rescale_cap_mode", "rescale_cap_value", "rescale_cap_quantile"):
            options.pop(key, None)
    for key in ("obs_num_samples", "obs_cg_iters"):
        options.pop(key, None)
    if strategy == "obs_compensated":
        damping = float(options.pop("obs_damping", 1e-3))
        max_samples = int(options.pop("obs_max_samples", 25_000))
        compensation_mode = str(options.pop("obs_compensation_mode", "diagonal"))
        exact_block_threshold = int(options.pop("obs_exact_block_threshold", 128))
        prune_kwargs.update({
            "damping": damping,
            "max_samples": max_samples,
            "compensation_mode": compensation_mode,
            "exact_block_threshold": exact_block_threshold,
        })
        prune_meta.update({
            "obs_damping": damping,
            "obs_max_samples": max_samples,
            "obs_compensation_mode": compensation_mode,
            "obs_exact_block_threshold": exact_block_threshold,
        })
    else:
        for key in ("obs_max_samples", "obs_compensation_mode", "obs_exact_block_threshold"):
            options.pop(key, None)
    return prune_kwargs, prune_meta


def _coerce_kwargs(payload: Any, label: str) -> Dict[str, Any]:
    if payload is None:
        return {}
    if isinstance(payload, str):
        try:
            return json.loads(payload)
        except json.JSONDecodeError as exc:  # pragma: no cover - defensive
            raise ValueError(f"{label} must be JSON-serialisable: {payload}") from exc
    if isinstance(payload, dict):
        return dict(payload)
    raise TypeError(f"{label} must be a dict or JSON string, got {type(payload)}")


# ---------------------------------------------------------------------------
# Seeding helpers and lightweight dataset/model factories
# ---------------------------------------------------------------------------


@contextmanager
def temporary_seed(seed: Optional[int]):
    if seed is None:
        yield
        return
    np_state = np.random.get_state()
    py_state = random.getstate()
    torch_state = torch.random.get_rng_state()
    if torch.cuda.is_available():
        cuda_states = torch.cuda.get_rng_state_all()
    else:
        cuda_states = None
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        yield
    finally:
        np.random.set_state(np_state)
        random.setstate(py_state)
        torch.random.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


def fresh_model(
    input_dim=3,
    hidden_size=128,
    output_dim=2,
    device="cpu",
    *,
    model_type: str = "ctrnn",
    **model_kwargs,
):
    model_type = model_type.lower()
    if model_type == "ctrnn":
        activation = model_kwargs.get("activation", "tanh")
        return CTRNN(
            input_dim=input_dim,
            hidden_size=hidden_size,
            output_dim=output_dim,
            dt=10,
            tau=100,
            activation=activation,
            preact_noise=0.0,
            postact_noise=0.0,
            use_dale=model_kwargs.get("use_dale", False),
            ei_ratio=model_kwargs.get("ei_ratio", 0.8),
            no_self_connections=model_kwargs.get("no_self_connections", True),
            recurrent_init_std=model_kwargs.get("recurrent_init_std", None),
            recurrent_bias_init=model_kwargs.get("recurrent_bias_init", "zero"),
            recurrent_bias_constant=model_kwargs.get("recurrent_bias_constant", 0.0),
            scaling=1.0,
            bias=True,
        ).to(device)
    raise ValueError("This release supports model_type='ctrnn' only.")


def append_results_csv(results_list: Iterable[Dict[str, Any]], csv_path: str = "results.csv"):
    import csv

    rows = list(results_list)
    if not rows:
        return
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)

    existing_rows: List[Dict[str, Any]] = []
    existing_fields: List[str] = []
    if os.path.exists(csv_path):
        with open(csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            existing_rows = list(reader)
            existing_fields = reader.fieldnames or []

    all_rows = existing_rows + rows
    keys = sorted({k for row in all_rows for k in row.keys() if k is not None})

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()

        def _sanitize(row: Dict[str, Any]) -> Dict[str, Any]:
            sanitized = {}
            for key in keys:
                value = row.get(key, "")
                if isinstance(value, (dict, list)):
                    sanitized[key] = json.dumps(value, sort_keys=True)
                else:
                    sanitized[key] = value
            return sanitized

        for r in existing_rows:
            writer.writerow(_sanitize(r))
        for r in rows:
            writer.writerow(_sanitize(r))


def evaluate_on_fixed_batches(
    model: torch.nn.Module,
    batches,
    criterion,
    *,
    dataset_last_only: bool,
    eval_last_only: bool | None,
) -> Dict[str, float]:
    prev_mode = model.training
    model.eval()
    if eval_last_only is None:
        eval_last_only = dataset_last_only
    total_loss = 0.0
    total_loss_weight = 0
    total_decision_correct = 0
    total_decision_count = 0
    total_seq_correct = 0
    total_seq_count = 0
    total_last_valid_correct = 0
    total_last_valid_count = 0
    total_last_valid_loss = 0.0
    total_last_valid_loss_weight = 0
    with torch.no_grad():
        for x_batch, y_batch in batches:
            logits, _ = model(x_batch)
            decision_logits = logits[-1]
            decision_targets = y_batch[-1]
            decision_valid = decision_targets >= 0
            decision_N = int(decision_valid.sum().item())
            decision_loss = criterion(decision_logits, decision_targets)

            if eval_last_only:
                loss_val = decision_loss
                loss_weight = decision_N
            else:
                seq_logits = logits.view(-1, logits.size(-1))
                seq_targets = y_batch.view(-1)
                loss_val = criterion(seq_logits, seq_targets)
                loss_weight = int((seq_targets >= 0).sum().item())

            if loss_weight > 0:
                total_loss += float(loss_val) * loss_weight
                total_loss_weight += loss_weight

            decision_pred = decision_logits.argmax(dim=-1)
            if decision_N > 0:
                decision_correct = ((decision_pred == decision_targets) & decision_valid).sum().item()
                total_decision_correct += int(decision_correct)
                total_decision_count += int(decision_N)

            seq_pred = logits.argmax(dim=-1)
            seq_valid = y_batch >= 0
            seq_total = int(seq_valid.sum().item())
            if seq_total > 0:
                seq_correct = ((seq_pred == y_batch) & seq_valid).sum().item()
                total_seq_correct += int(seq_correct)
                total_seq_count += int(seq_total)

            last_logits, last_targets = last_valid_logits_targets(logits, y_batch)
            if last_logits is not None and last_targets is not None:
                last_loss = criterion(last_logits, last_targets)
                last_pred = last_logits.argmax(dim=-1)
                last_count = int(last_targets.numel())
                total_last_valid_loss += float(last_loss) * last_count
                total_last_valid_loss_weight += last_count
                total_last_valid_correct += int((last_pred == last_targets).sum().item())
                total_last_valid_count += last_count
    if prev_mode:
        model.train()
    mean_loss = total_loss / max(1, total_loss_weight)
    decision_acc = total_decision_correct / max(1, total_decision_count)
    sequence_acc = total_seq_correct / max(1, total_seq_count)
    last_valid_acc = total_last_valid_correct / max(1, total_last_valid_count)
    last_valid_loss = total_last_valid_loss / max(1, total_last_valid_loss_weight)
    return {
        "loss": mean_loss,
        "acc": decision_acc,
        "acc_sequence": sequence_acc,
        "acc_last_valid": last_valid_acc,
        "loss_last_valid": last_valid_loss,
    }


def _seed_data_source(data: Any, seed: Optional[int]) -> None:
    """Best-effort reseeding for dataset objects with their own RNG state."""
    if seed is None:
        return
    seed = int(seed)
    dataset = getattr(data, "dataset", None)
    if dataset is not None and hasattr(dataset, "seed"):
        try:
            dataset.seed(seed)
            return
        except Exception:
            pass
    env = getattr(data, "env", None)
    if env is not None:
        try:
            if hasattr(env, "seed"):
                env.seed(seed)
        except Exception:
            pass
        try:
            env.reset(seed=seed)
        except TypeError:
            try:
                if hasattr(env, "seed"):
                    env.seed(seed)
                else:
                    env.reset()
            except Exception:
                pass
        action_space = getattr(env, "action_space", None)
        if hasattr(action_space, "seed"):
            try:
                action_space.seed(seed)
            except Exception:
                pass
    envs = getattr(data, "envs", None)
    if envs is not None:
        for idx, env_i in enumerate(envs):
            env_seed = seed + idx
            try:
                if hasattr(env_i, "seed"):
                    env_i.seed(env_seed)
            except Exception:
                pass
            try:
                env_i.reset(seed=env_seed)
            except TypeError:
                try:
                    if hasattr(env_i, "seed"):
                        env_i.seed(env_seed)
                    else:
                        env_i.reset()
                except Exception:
                    pass
            action_space = getattr(env_i, "action_space", None)
            if hasattr(action_space, "seed"):
                try:
                    action_space.seed(env_seed)
                except Exception:
                    pass


def _load_batch_list(path: Path, device: str) -> List[tuple[torch.Tensor, torch.Tensor]]:
    try:
        raw = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        raw = torch.load(path, map_location=device)
    return [(x.to(device), y.to(device)) for x, y in raw]


def _save_batch_list(path: Path, batches: List[tuple[torch.Tensor, torch.Tensor]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cpu_batches = [(x.detach().cpu(), y.detach().cpu()) for x, y in batches]
    torch.save(cpu_batches, path)


# ---------------------------------------------------------------------------
# Experiment driver
# ---------------------------------------------------------------------------


def run_prune_experiment(
    strategy,
    amount,
    train_steps=600,
    ft_steps=200,
    last_only=True,
    model_type="ctrnn",
    eval_last_only=None,
    seed=0,
    device="cpu",
    movement_batches=20,
    base_model=None,
    task: str = "modcog:ctxdlydm1seql",
    no_prune: bool = False,
    run_id: Optional[str] = None,
    return_model: bool = False,
    **kwargs,
):
    save_model_path = kwargs.pop("save_model_path", None)
    load_model_path = kwargs.pop("load_model_path", None)
    skip_training = bool(kwargs.pop("skip_training", False))
    eval_seed_base = kwargs.pop("eval_seed", None)
    eval_sample_batches = int(kwargs.pop("eval_sample_batches", 0))
    eval_batches_path_raw = kwargs.pop("eval_batches_path", None)
    eval_batches_path = None if eval_batches_path_raw in {None, ""} else Path(str(eval_batches_path_raw))
    eval_steps_pre0 = kwargs.pop("eval_steps_pre0", 50)
    eval_steps_pre = kwargs.pop("eval_steps_pre", 100)
    eval_steps_post0 = kwargs.pop("eval_steps_post0", 100)
    eval_steps_post = kwargs.pop("eval_steps_post", 100)
    lr = float(kwargs.pop("lr", 1e-3))
    lr_start_raw = kwargs.pop("lr_start", None)
    lr_start = None if lr_start_raw is None else float(lr_start_raw)
    lr_warmup_steps = int(kwargs.pop("lr_warmup_steps", 0))
    adaptive_lr = bool(kwargs.pop("adaptive_lr", False))
    adaptive_lr_min_raw = kwargs.pop("adaptive_lr_min", None)
    adaptive_lr_min = None if adaptive_lr_min_raw is None else float(adaptive_lr_min_raw)
    adaptive_lr_increase_factor = float(kwargs.pop("adaptive_lr_increase_factor", 1.03))
    adaptive_lr_decrease_factor = float(kwargs.pop("adaptive_lr_decrease_factor", 0.5))
    adaptive_lr_patience = int(kwargs.pop("adaptive_lr_patience", 25))
    adaptive_lr_min_delta = float(kwargs.pop("adaptive_lr_min_delta", 1e-4))
    adaptive_lr_smoothing = float(kwargs.pop("adaptive_lr_smoothing", 0.05))
    adaptive_lr_window = int(kwargs.pop("adaptive_lr_window", 50))
    adaptive_lr_improve_fraction = float(kwargs.pop("adaptive_lr_improve_fraction", 0.6))
    clip = float(kwargs.pop("clip", 1.0))
    recurrent_l2_lambda = float(kwargs.pop("recurrent_l2_lambda", 0.0))
    train_progress = bool(kwargs.pop("train_progress", False))
    train_progress_every = int(kwargs.pop("train_progress_every", 100))
    train_select_best = bool(kwargs.pop("train_select_best", False))
    train_val_interval = int(kwargs.pop("train_val_interval", 0))
    train_best_metric = str(kwargs.pop("train_best_metric", "acc_sequence"))
    train_best_metric_mode = str(kwargs.pop("train_best_metric_mode", "max"))
    train_best_tie_metric_raw = kwargs.pop("train_best_tie_metric", "loss_last_valid")
    train_best_tie_metric = None if train_best_tie_metric_raw in {None, ""} else str(train_best_tie_metric_raw)
    train_best_tie_metric_mode = str(kwargs.pop("train_best_tie_metric_mode", "min"))
    train_history_path_raw = kwargs.pop("train_history_path", None)
    hidden_size_override = kwargs.pop("hidden_size", None)
    model_kwargs = {}
    for key in (
        "use_dale",
        "ei_ratio",
        "no_self_connections",
        "activation",
        "recurrent_init_std",
        "recurrent_bias_init",
        "recurrent_bias_constant",
    ):
        if key in kwargs:
            model_kwargs[key] = kwargs.pop(key)

    ng_kwargs_raw = kwargs.pop("ng_kwargs", None)
    ng_T = kwargs.pop("ng_T", None)
    ng_B = kwargs.pop("ng_B", None)
    ng_dataset_kwargs_raw = kwargs.pop("ng_dataset_kwargs", None)
    score_batch_seed_raw = kwargs.pop("score_batch_seed", None)
    score_batch_seed = None if score_batch_seed_raw is None else int(score_batch_seed_raw)
    score_batches_path_raw = kwargs.pop("score_batches_path", None)
    score_batches_path = None if score_batches_path_raw in {None, ""} else Path(str(score_batches_path_raw))
    score_batch_max_resamples = int(kwargs.pop("score_batch_max_resamples", 10) or 10)
    score_batch_min_valid = int(kwargs.pop("score_batch_min_valid", 1) or 1)
    source_metadata = {}
    for key in (
        "source_model_label",
        "source_run_id",
        "source_recurrent_l2_lambda",
        "source_network_seed",
        "pruning_seed",
    ):
        if key in kwargs:
            source_metadata[key] = kwargs.pop(key)

    prune_phase = kwargs.pop("prune_phase", "post")
    if prune_phase not in {"pre", "post"}:
        raise ValueError("prune_phase must be 'pre' or 'post'")
    prune_kwargs, prune_meta = _extract_prune_kwargs(strategy, kwargs)
    if kwargs:
        unknown = ", ".join(sorted(kwargs.keys()))
        raise ValueError(f"Unsupported keyword arguments for run_prune_experiment: {unknown}")

    resolved_run_id = run_id or make_run_id()
    config = ExperimentConfig(
        strategy=strategy,
        amount=float(amount),
        train_steps=int(train_steps),
        ft_steps=int(ft_steps),
        last_only=bool(last_only),
        model_type=str(model_type),
        seed=int(seed),
        device=device,
        movement_batches=int(movement_batches),
        task=task,
        no_prune=bool(no_prune),
        prune_phase=prune_phase,
        run_id=resolved_run_id,
    )

    pruned = not config.no_prune and config.strategy != "none"
    normalized_amount = validate_prune_fraction(config.amount) if pruned else 0.0
    config.amount = normalized_amount

    if eval_sample_batches > 0 and eval_seed_base is None:
        eval_seed_base = config.seed

    if eval_last_only is None:
        eval_last_only = config.last_only

    set_global_seed(config.seed)
    run_dir = _ensure_run_directory(config.run_id)
    if train_select_best and train_val_interval <= 0:
        train_val_interval = 500
    if train_best_metric_mode not in {"max", "min"}:
        raise ValueError("train_best_metric_mode must be 'max' or 'min'")
    if train_best_tie_metric_mode not in {"max", "min"}:
        raise ValueError("train_best_tie_metric_mode must be 'max' or 'min'")
    train_history_path = (
        Path(train_history_path_raw)
        if train_history_path_raw
        else run_dir / "training_history.csv"
    )

    # ------------------------------------------------------------------
    # Build dataset/task
    # ------------------------------------------------------------------
    task_meta: Dict[str, Any] = {"task": task, "last_only_eval": bool(last_only)}
    env_kwargs: Dict[str, Any] = {}
    uses_modcog = task.startswith("modcog:")
    env_kwargs = _coerce_kwargs(ng_kwargs_raw, "ng_kwargs") if uses_modcog else None
    dataset_kwargs = _coerce_kwargs(ng_dataset_kwargs_raw, "ng_dataset_kwargs") if uses_modcog else None
    if uses_modcog and last_only:
        raise ValueError("Mod-Cog tasks do not support last_only=True; use full-sequence training/eval.")

    if uses_modcog:
        env_suffix = task.split("modcog:", 1)[1].strip()
        if not env_suffix:
            raise ValueError("Mod_Cog task identifier missing after 'modcog:' prefix.")
        env_kwargs_copy = dict(env_kwargs or {})
        dataset_kwargs = dict(dataset_kwargs or {})
        try:
            builder_info = resolve_modcog_callable(env_suffix)
        except ImportError as exc:
            raise ImportError(
                "Mod_Cog tasks requested but the vendored Mod_Cog package could not be imported."
            ) from exc
        T = int(ng_T) if ng_T is not None else 400
        B = int(ng_B) if ng_B is not None else 64
        if builder_info is None:
            raise ValueError(f"Unknown Mod-Cog task builder: {env_suffix!r}")
        canonical_name, builder_fn = builder_info
        env_id = f"Mod_Cog-{canonical_name}-v0"
        env_label = canonical_name
        dataset_backend = "mod_cog_builder"
        dataset_env_source = builder_fn(**env_kwargs_copy)
        dataset_env_kwargs = None
        data = ModCogTrialDM(
            dataset_env_source,
            T=T,
            B=B,
            device=device,
            last_only=last_only,
            seed=config.seed,
            env_kwargs=dataset_env_kwargs,
            mask_fixation=True,
        )
        task_meta.update({
            "env": env_label,
            "env_id": env_id,
            "T": T,
            "B": B,
            "dataset_last_only": bool(last_only),
            "env_kwargs": env_kwargs_copy,
            "backend": dataset_backend,
        })
        input_dim = data.input_dim
        output_dim = data.n_classes
    else:
        raise ValueError("This release supports task='modcog:<task>' only.")

    if input_dim is None or output_dim is None:
        X_tmp, Y_tmp = data.sample_batch()
        input_dim = X_tmp.size(-1)
        output_dim = int(max(2, int(Y_tmp.max().item()) + 1))

    fixed_batches = None
    dataset_last_only_flag = bool(getattr(data, "last_only", last_only))

    if eval_batches_path is not None and eval_batches_path.exists():
        fixed_batches = _load_batch_list(eval_batches_path, device)
    elif eval_sample_batches > 0:
        fixed_batches = []
        seed_offset = 9
        base_seed = None if eval_seed_base is None else int(eval_seed_base) * 10 + seed_offset
        with temporary_seed(base_seed):
            _seed_data_source(data, base_seed)
            for _ in range(eval_sample_batches):
                x_batch, y_batch = data.sample_batch()
                fixed_batches.append((x_batch.to(device), y_batch.to(device)))
            _seed_data_source(data, config.seed)
        if eval_batches_path is not None:
            _save_batch_list(eval_batches_path, fixed_batches)

    state_dict_cached = None
    def _infer_hidden_from_state(state_dict: Dict[str, torch.Tensor]) -> Optional[int]:
        hh_weight = state_dict.get("hidden_layer.weight")
        if hh_weight is not None:
            return int(hh_weight.shape[0])
        return None

    if load_model_path is not None:
        try:
            state_dict_cached = torch.load(load_model_path, map_location=device, weights_only=True)
        except TypeError:
            state_dict_cached = torch.load(load_model_path, map_location=device)
        if hidden_size_override is None and state_dict_cached is not None:
            inferred = _infer_hidden_from_state(state_dict_cached)
            if inferred is not None:
                hidden_size_override = inferred

    # ------------------------------------------------------------------
    # Build model and optimiser
    # ------------------------------------------------------------------
    if base_model is None:
        hidden_size = hidden_size_override or 128
        model = fresh_model(
            input_dim=input_dim,
            hidden_size=int(hidden_size),
            output_dim=output_dim,
            device=device,
            model_type=model_type,
            **model_kwargs,
        )
    else:
        model = deepcopy(base_model).to(device)

    if load_model_path is not None:
        state = state_dict_cached
        if state is None:
            try:
                state = torch.load(load_model_path, map_location=device, weights_only=True)
            except TypeError:
                state = torch.load(load_model_path, map_location=device)
        model.load_state_dict(state)

    enforce_constraints(model)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    pruner = get_pruner(strategy) if pruned else None

    def _batch_valid_count(yb: torch.Tensor, use_last_only: bool) -> int:
        if use_last_only:
            return int((yb[-1] != -1).sum().item())
        return int((yb != -1).sum().item())

    def sample_batches(num: int | None):
        if not num or num <= 0:
            return None
        if score_batches_path is not None and score_batches_path.exists():
            return _load_batch_list(score_batches_path, device)

        def _draw() -> List[tuple[torch.Tensor, torch.Tensor]]:
            batches: List[tuple[torch.Tensor, torch.Tensor]] = []
            for _ in range(num):
                chosen_x = None
                chosen_y = None
                attempts = max(1, score_batch_max_resamples)
                for _attempt in range(attempts):
                    xb, yb = data.sample_batch()
                    if _batch_valid_count(yb, last_only) >= score_batch_min_valid:
                        chosen_x, chosen_y = xb, yb
                        break
                    if chosen_x is None:
                        # Keep first draw as a fallback to avoid infinite loops.
                        chosen_x, chosen_y = xb, yb
                batches.append((chosen_x.to(device), chosen_y.to(device)))
            return batches

        if score_batch_seed is None:
            batches = _draw()
            if score_batches_path is not None:
                _save_batch_list(score_batches_path, batches)
            return batches
        with temporary_seed(score_batch_seed):
            _seed_data_source(data, score_batch_seed)
            batches = _draw()
            _seed_data_source(data, config.seed)
            if score_batches_path is not None:
                _save_batch_list(score_batches_path, batches)
            return batches

    pre_prune_stats: Dict[str, Any] = {}
    pruned_pretraining = False
    if pruned and prune_phase == "pre":
        if skip_training or load_model_path is not None:
            raise ValueError("Cannot use prune_phase='pre' with pre-trained checkpoints or skip_training=True.")
        if pruner is None or not getattr(pruner, "supports_pretrain", False):
            raise ValueError(f"Strategy '{strategy}' does not support prune_phase='pre'.")
        batch_count = pruner.resolved_batch_count(movement_batches)
        score_batches = sample_batches(batch_count)
        context = PruneContext(
            model=model,
            amount=normalized_amount,
            criterion=criterion,
            last_only=last_only,
            device=device,
            batches=score_batches,
            metadata={"phase": "pre", "run_id": config.run_id},
        )
        stats = pruner.pretrain(context, **prune_kwargs)
        pre_prune_stats = dict(stats) if stats else {}
        pruned_pretraining = True

    def run_eval(steps: int, offset: int) -> Tuple[float, float]:
        if fixed_batches is not None:
            return evaluate_on_fixed_batches(
                model,
                fixed_batches,
                criterion,
                dataset_last_only=dataset_last_only_flag,
                eval_last_only=eval_last_only,
            )
        seed_val = None if eval_seed_base is None else int(eval_seed_base) * 10 + offset
        with temporary_seed(seed_val):
            return evaluate(
                model,
                data,
                device,
                criterion,
                steps=steps,
                dataset_last_only=dataset_last_only_flag,
                eval_last_only=eval_last_only,
            )

    # ------------------------------------------------------------------
    # Phase A: baseline evaluation
    # ------------------------------------------------------------------
    pre0_metrics = run_eval(eval_steps_pre0, 0)
    pre0_loss = pre0_metrics["loss"]
    pre0_acc = pre0_metrics["acc"]
    pre0_snapshot = snapshot_model(model)
    training_history: List[Dict[str, Any]] = []
    best_train_state: Dict[str, torch.Tensor] | None = None
    best_train_eval_metrics: Dict[str, Any] | None = None
    best_train_candidate: Dict[str, Any] | None = None
    best_train_step: Optional[int] = None
    best_train_restored = False

    def _record_training_validation(step: int, train_loss_value: float | None, metrics: Dict[str, Any]) -> None:
        nonlocal best_train_state, best_train_eval_metrics, best_train_candidate, best_train_step
        row: Dict[str, Any] = {
            "run_id": resolved_run_id,
            "phase": "train",
            "step": int(step),
            "train_recent_loss": train_loss_value,
        }
        for key, value in metrics.items():
            row[f"val_{key}"] = value
        training_history.append(row)
        candidate = dict(metrics)
        candidate["step"] = int(step)
        candidate["train_recent_loss"] = train_loss_value
        if _metric_better(
            candidate,
            best_train_candidate,
            metric=train_best_metric,
            mode=train_best_metric_mode,
            tie_metric=train_best_tie_metric,
            tie_mode=train_best_tie_metric_mode,
        ):
            best_train_candidate = dict(candidate)
            best_train_eval_metrics = dict(metrics)
            best_train_step = int(step)
            best_train_state = _clone_state_dict_cpu(model)

    # ------------------------------------------------------------------
    # Phase B: baseline training
    # ------------------------------------------------------------------
    if not skip_training and train_steps > 0:
        if train_select_best:
            _record_training_validation(0, None, pre0_metrics)
            completed_steps = 0
            total_train_steps = int(train_steps)
            training_base_lrs = [float(group.get("lr", 0.0)) for group in opt.param_groups]
            while completed_steps < total_train_steps:
                chunk_steps = min(int(train_val_interval), total_train_steps - completed_steps)
                recent_train_loss = train_epoch(
                    model,
                    data,
                    device,
                    opt,
                    criterion,
                    steps=chunk_steps,
                    last_only=last_only,
                    clip=clip,
                    progress=train_progress,
                    progress_label=f"{resolved_run_id}:train",
                    progress_every=train_progress_every,
                    progress_step_offset=completed_steps,
                    progress_total_steps=total_train_steps,
                    lr_start=lr_start if (lr_start is not None and completed_steps < int(lr_warmup_steps)) else None,
                    lr_warmup_steps=lr_warmup_steps,
                    lr_warmup_step_offset=completed_steps,
                    lr_base_lrs=training_base_lrs,
                    adaptive_lr=adaptive_lr,
                    adaptive_lr_min=adaptive_lr_min,
                    adaptive_lr_increase_factor=adaptive_lr_increase_factor,
                    adaptive_lr_decrease_factor=adaptive_lr_decrease_factor,
                    adaptive_lr_patience=adaptive_lr_patience,
                    adaptive_lr_min_delta=adaptive_lr_min_delta,
                    adaptive_lr_smoothing=adaptive_lr_smoothing,
                    adaptive_lr_window=adaptive_lr_window,
                    adaptive_lr_improve_fraction=adaptive_lr_improve_fraction,
                    recurrent_l2_lambda=recurrent_l2_lambda,
                )
                completed_steps += chunk_steps
                validation_metrics = run_eval(eval_steps_pre, 1)
                _record_training_validation(completed_steps, recent_train_loss, validation_metrics)
            if best_train_state is not None:
                best_train_restored = best_train_step != total_train_steps
                model.load_state_dict(best_train_state)
            pre_metrics = dict(best_train_eval_metrics) if best_train_eval_metrics is not None else run_eval(eval_steps_pre, 1)
        else:
            train_epoch(
                model,
                data,
                device,
                opt,
                criterion,
                steps=train_steps,
                last_only=last_only,
                clip=clip,
                progress=train_progress,
                progress_label=f"{resolved_run_id}:train",
                progress_every=train_progress_every,
                lr_start=lr_start,
                lr_warmup_steps=lr_warmup_steps,
                adaptive_lr=adaptive_lr,
                adaptive_lr_min=adaptive_lr_min,
                adaptive_lr_increase_factor=adaptive_lr_increase_factor,
                adaptive_lr_decrease_factor=adaptive_lr_decrease_factor,
                adaptive_lr_patience=adaptive_lr_patience,
                adaptive_lr_min_delta=adaptive_lr_min_delta,
                adaptive_lr_smoothing=adaptive_lr_smoothing,
                adaptive_lr_window=adaptive_lr_window,
                adaptive_lr_improve_fraction=adaptive_lr_improve_fraction,
                recurrent_l2_lambda=recurrent_l2_lambda,
            )
            pre_metrics = run_eval(eval_steps_pre, 1)
        pre_loss = pre_metrics["loss"]
        pre_acc = pre_metrics["acc"]
        pre_snapshot = snapshot_model(model)
    else:
        pre_metrics = dict(pre0_metrics)
        pre_loss, pre_acc = pre0_loss, pre0_acc
        pre_snapshot = dict(pre0_snapshot)

    prune_stats_post: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Phase C: pruning
    # ------------------------------------------------------------------
    if pruned and prune_phase == "post":
        batch_count = pruner.resolved_batch_count(movement_batches)
        score_batches: Optional[List[tuple[torch.Tensor, torch.Tensor]]] = None
        if batch_count > 0:
            score_batches = sample_batches(batch_count)

        context = PruneContext(
            model=model,
            amount=normalized_amount,
            criterion=criterion,
            last_only=last_only,
            device=device,
            batches=score_batches,
            metadata={"phase": "post", "run_id": config.run_id},
        )
        prune_stats_post = pruner.run(context, **prune_kwargs)
        post0_metrics = run_eval(eval_steps_post0, 2)
        post0_loss = post0_metrics["loss"]
        post0_acc = post0_metrics["acc"]
        post0_snapshot = snapshot_model(model)
    else:
        post0_metrics = dict(pre_metrics)
        post0_loss, post0_acc = pre_loss, pre_acc
        post0_snapshot = dict(pre_snapshot)

    # ------------------------------------------------------------------
    # Phase D: optional fine-tuning
    # ------------------------------------------------------------------
    if ft_steps > 0:
        train_epoch(
            model,
            data,
            device,
            opt,
            criterion,
            steps=ft_steps,
            last_only=last_only,
            clip=clip,
            progress=train_progress,
            progress_label=f"{resolved_run_id}:finetune",
            progress_every=train_progress_every,
            lr_start=lr_start,
            lr_warmup_steps=lr_warmup_steps,
            adaptive_lr=adaptive_lr,
            adaptive_lr_min=adaptive_lr_min,
            adaptive_lr_increase_factor=adaptive_lr_increase_factor,
            adaptive_lr_decrease_factor=adaptive_lr_decrease_factor,
            adaptive_lr_patience=adaptive_lr_patience,
            adaptive_lr_min_delta=adaptive_lr_min_delta,
            adaptive_lr_smoothing=adaptive_lr_smoothing,
            adaptive_lr_window=adaptive_lr_window,
            adaptive_lr_improve_fraction=adaptive_lr_improve_fraction,
            recurrent_l2_lambda=recurrent_l2_lambda,
        )
        post_metrics = run_eval(eval_steps_post, 3)
        post_loss = post_metrics["loss"]
        post_acc = post_metrics["acc"]
        post_snapshot = snapshot_model(model)
    else:
        # When no fine-tuning occurs, the post-prune model is identical to post0.
        post_metrics = dict(post0_metrics)
        post_loss = post0_loss
        post_acc = post0_acc
        post_snapshot = dict(post0_snapshot)

    finalize_pruning(model)

    if training_history:
        _write_training_history(train_history_path, training_history)

    if save_model_path is not None:
        os.makedirs(os.path.dirname(save_model_path) or ".", exist_ok=True)
        torch.save(model.state_dict(), save_model_path, _use_new_zipfile_serialization=True)

    # ------------------------------------------------------------------
    # Assemble metrics and config snapshots
    # ------------------------------------------------------------------
    phase_metrics = {
        "pre0": {**pre0_snapshot, "loss": pre0_loss, "acc": pre0_acc},
        "pre": {**pre_snapshot, "loss": pre_loss, "acc": pre_acc},
        "post0": {**post0_snapshot, "loss": post0_loss, "acc": post0_acc},
        "post": {**post_snapshot, "loss": post_loss, "acc": post_acc},
    }
    # merge additional evaluation metrics (e.g., sequence accuracies)
    for name, metrics in (
        ("pre0", pre0_metrics),
        ("pre", pre_metrics),
        ("post0", post0_metrics),
        ("post", post_metrics),
    ):
        for key, value in metrics.items():
            if key in {"loss", "acc"}:
                continue
            phase_metrics[name][key] = value

    extras = {
        "delta_post0_acc": pre_acc - post0_acc,
        "delta_post_acc": pre_acc - post_acc,
        "pruned": bool(pruned),
        "amount": normalized_amount,
        "amount_step": PRUNE_AMOUNT_STEP,
        "ft_steps": ft_steps,
        "train_steps": train_steps,
        "prune_phase": prune_phase,
        "pruned_pretraining": pruned_pretraining,
        "train_select_best": bool(train_select_best),
        "train_best_step": best_train_step,
        "train_best_restored": bool(best_train_restored),
    }
    if best_train_candidate is not None:
        for key, value in best_train_candidate.items():
            if key in {"step", "train_recent_loss"}:
                extras[f"train_best_{key}"] = value
            else:
                extras[f"train_best_val_{key}"] = value
        extras["train_history_csv"] = str(train_history_path)
    metrics_report = compile_run_metrics(phase_metrics, extras=extras)
    metrics_path = save_metrics(run_dir, metrics_report)

    model_meta = {
        "class": type(model).__name__,
        "model_type": model_type,
        "input_dim": getattr(model, "I", None),
        "hidden_size": getattr(model, "H", None),
        "output_dim": getattr(model, "O", None),
        "alpha": getattr(model, "alpha", None),
        "activation": getattr(model, "_activation_name", None),
        "preact_noise": getattr(model, "preact_noise", None),
        "postact_noise": getattr(model, "postact_noise", None),
        "use_dale": bool(getattr(model, "use_dale", False)),
        "no_self_connections": bool(getattr(model, "no_self_connections", False)),
    }

    config_metadata = config.to_metadata()
    config_metadata.update({
        "skip_training": bool(skip_training),
        "eval_seed": eval_seed_base,
        "eval_sample_batches": eval_sample_batches,
        "eval_batches_path": str(eval_batches_path) if eval_batches_path is not None else None,
        "eval_steps_pre0": eval_steps_pre0,
        "eval_steps_pre": eval_steps_pre,
        "eval_steps_post0": eval_steps_post0,
        "eval_steps_post": eval_steps_post,
        "eval_last_only": bool(eval_last_only),
        "lr": lr,
        "lr_start": lr_start,
        "lr_warmup_steps": lr_warmup_steps,
        "adaptive_lr": adaptive_lr,
        "adaptive_lr_min": adaptive_lr_min,
        "adaptive_lr_increase_factor": adaptive_lr_increase_factor,
        "adaptive_lr_decrease_factor": adaptive_lr_decrease_factor,
        "adaptive_lr_patience": adaptive_lr_patience,
        "adaptive_lr_min_delta": adaptive_lr_min_delta,
        "adaptive_lr_smoothing": adaptive_lr_smoothing,
        "adaptive_lr_window": adaptive_lr_window,
        "adaptive_lr_improve_fraction": adaptive_lr_improve_fraction,
        "clip": clip,
        "recurrent_l2_lambda": recurrent_l2_lambda,
        "train_progress": train_progress,
        "train_progress_every": train_progress_every,
        "train_select_best": train_select_best,
        "train_val_interval": train_val_interval,
        "train_best_metric": train_best_metric,
        "train_best_metric_mode": train_best_metric_mode,
        "train_best_tie_metric": train_best_tie_metric,
        "train_best_tie_metric_mode": train_best_tie_metric_mode,
        "train_history_path": str(train_history_path) if training_history else None,
        "hidden_size_override": hidden_size_override,
        "movement_batches": movement_batches,
        "model_kwargs": model_kwargs,
        "ng_T": task_meta.get("T", ng_T),
        "ng_B": task_meta.get("B", ng_B),
        "ng_kwargs": ng_kwargs_raw,
        "ng_dataset_kwargs": ng_dataset_kwargs_raw,
        "score_batch_seed": score_batch_seed,
        "score_batches_path": str(score_batches_path) if score_batches_path is not None else None,
        "score_batch_max_resamples": score_batch_max_resamples,
        "score_batch_min_valid": score_batch_min_valid,
    })
    config_metadata.update(source_metadata)
    if strategy in {
        "noise_prune",
        "noise_prune_capped_rescale",
        "vanilla_mask_only",
        "simulation_noise_prune_mask_only",
        "simulation_noise_prune_rescale",
        "simulation_noise_prune_capped_rescale",
    }:
        config_metadata.update({
            "noise_sigma": prune_meta.get("sigma"),
            "noise_eps": prune_meta.get("eps"),
            "noise_leak_shift": prune_meta.get("leak_shift"),
            "noise_matched_diagonal": prune_meta.get("matched_diagonal"),
            "noise_rng_seed": prune_meta.get("rng_seed"),
            # Keep compatibility with existing analysis scripts that read prune_* columns.
            "prune_sigma": prune_meta.get("sigma"),
            "prune_eps": prune_meta.get("eps"),
            "prune_leak_shift": prune_meta.get("leak_shift"),
            "prune_matched_diagonal": prune_meta.get("matched_diagonal"),
            "prune_rng_seed": prune_meta.get("rng_seed"),
        })
        if strategy in {
            "simulation_noise_prune_mask_only",
            "simulation_noise_prune_rescale",
            "simulation_noise_prune_capped_rescale",
        }:
            config_metadata.update({
                "sim_np_sigma": prune_meta.get("sim_np_sigma"),
                "sim_np_sigma_source": prune_meta.get("sim_np_sigma_source"),
                "sim_np_observable_space": prune_meta.get("sim_np_observable_space"),
                "sim_np_inject_space": prune_meta.get("sim_np_inject_space"),
                "sim_np_centering": prune_meta.get("sim_np_centering"),
                "sim_np_max_samples": prune_meta.get("sim_np_max_samples"),
                "sim_np_burn_in_steps": prune_meta.get("sim_np_burn_in_steps"),
                "prune_sim_np_sigma": prune_meta.get("sim_np_sigma"),
                "prune_sim_np_sigma_source": prune_meta.get("sim_np_sigma_source"),
                "prune_sim_np_observable_space": prune_meta.get("sim_np_observable_space"),
                "prune_sim_np_inject_space": prune_meta.get("sim_np_inject_space"),
                "prune_sim_np_centering": prune_meta.get("sim_np_centering"),
                "prune_sim_np_max_samples": prune_meta.get("sim_np_max_samples"),
                "prune_sim_np_burn_in_steps": prune_meta.get("sim_np_burn_in_steps"),
            })
    if save_model_path is not None:
        config_metadata["save_model_path"] = save_model_path
    if load_model_path is not None:
        config_metadata["load_model_path"] = load_model_path

    config_snapshot = {
        "experiment": config_metadata,
        "task": task_meta,
        "model": model_meta,
        "pruning": {
            "strategy": strategy,
            "amount": normalized_amount,
            "step": PRUNE_AMOUNT_STEP,
            "applied": bool(pruned),
            "options": prune_meta,
        },
    }
    config_paths = _write_config_snapshot(run_dir, config_snapshot)

    row: Dict[str, Any] = {
        **config_metadata,
        **metrics_report,
        "run_dir": str(run_dir),
        "config_json": str(config_paths["json"]),
        "config_yaml": str(config_paths["yaml"]),
        "metrics_json": str(metrics_path),
    }
    combined_prune_stats = dict(pre_prune_stats)
    combined_prune_stats.update(prune_stats_post)
    if combined_prune_stats:
        for key, value in combined_prune_stats.items():
            row[f"prune_{key}"] = value

    if return_model:
        return row, model.cpu()
    return row


__all__ = ["append_results_csv", "fresh_model", "run_prune_experiment", "temporary_seed"]
