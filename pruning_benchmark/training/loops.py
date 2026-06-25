"""Training and evaluation loops shared across experiments."""

from __future__ import annotations

import sys
import time
from collections import deque

import torch
import torch.nn as nn

from ..pruning import enforce_constraints


def _format_seconds(seconds: float) -> str:
    seconds = max(0, int(seconds))
    minutes, secs = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:d}h{minutes:02d}m"
    if minutes:
        return f"{minutes:d}m{secs:02d}s"
    return f"{secs:d}s"


def _write_progress(
    label: str,
    step: int,
    steps: int,
    loss: float,
    started: float,
    *,
    lr: float | None = None,
    final: bool = False,
) -> None:
    width = 24
    frac = min(1.0, max(0.0, step / max(1, steps)))
    filled = int(round(width * frac))
    bar = "#" * filled + "-" * (width - filled)
    elapsed = time.time() - started
    eta = 0.0 if step <= 0 else elapsed * (steps - step) / max(1, step)
    end = "\n" if final else "\r"
    lr_text = "" if lr is None else f" lr={lr:.3g}"
    sys.stderr.write(
        f"{label} [{bar}] {step}/{steps} "
        f"loss={loss:.4g}{lr_text} elapsed={_format_seconds(elapsed)} eta={_format_seconds(eta)}"
        f"{' ' * 8}{end}"
    )
    sys.stderr.flush()


def last_valid_logits_targets(logits: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Return logits/targets at the last valid timestep for each batch element."""
    if targets.ndim != 2 or logits.ndim != 3:
        return None, None
    valid = targets >= 0
    has_valid = valid.any(dim=0)
    if not bool(has_valid.any().item()):
        return None, None
    time_idx = torch.arange(targets.size(0), device=targets.device).view(-1, 1).expand_as(targets)
    last_idx = torch.where(valid, time_idx, torch.zeros_like(time_idx)).max(dim=0).values
    batch_idx = torch.arange(targets.size(1), device=targets.device)
    keep = has_valid
    return logits[last_idx[keep], batch_idx[keep]], targets[last_idx[keep], batch_idx[keep]]


def train_epoch(
    model,
    data,
    device,
    opt,
    criterion,
    steps=50,
    last_only=True,
    clip=1.0,
    *,
    progress: bool = False,
    progress_label: str = "train",
    progress_every: int = 100,
    progress_step_offset: int = 0,
    progress_total_steps: int | None = None,
    lr_start: float | None = None,
    lr_warmup_steps: int = 0,
    lr_warmup_step_offset: int = 0,
    lr_base_lrs: list[float] | None = None,
    adaptive_lr: bool = False,
    adaptive_lr_min: float | None = None,
    adaptive_lr_increase_factor: float = 1.03,
    adaptive_lr_decrease_factor: float = 0.5,
    adaptive_lr_patience: int = 25,
    adaptive_lr_min_delta: float = 1e-4,
    adaptive_lr_smoothing: float = 0.05,
    adaptive_lr_window: int = 50,
    adaptive_lr_improve_fraction: float = 0.6,
    recurrent_l2_lambda: float = 0.0,
):
    model.train()
    total_loss, total_count = 0.0, 0
    recent_loss, recent_count = 0.0, 0
    started = time.time()
    progress_every = max(1, int(progress_every))
    progress_step_offset = max(0, int(progress_step_offset))
    progress_total_steps = int(progress_total_steps) if progress_total_steps is not None else int(steps)
    progress_total_steps = max(1, progress_total_steps)
    base_lrs = list(lr_base_lrs) if lr_base_lrs is not None else [float(group.get("lr", 0.0)) for group in opt.param_groups]
    use_adaptive_lr = bool(adaptive_lr)
    lr_warmup_step_offset = max(0, int(lr_warmup_step_offset))
    use_lr_warmup = (
        (not use_adaptive_lr)
        and lr_start is not None
        and int(lr_warmup_steps) > 0
        and lr_warmup_step_offset < int(lr_warmup_steps)
    )
    lr_warmup_steps = max(1, int(lr_warmup_steps)) if use_lr_warmup else 0
    adaptive_lr_max = base_lrs[0] if base_lrs else 0.0
    adaptive_lr_floor = float(adaptive_lr_min) if adaptive_lr_min is not None else (
        float(lr_start) if lr_start is not None else adaptive_lr_max
    )
    current_lr = None
    smoothed_loss = None
    best_smoothed_loss = None
    improvement_streak = 0
    recent_improvements: deque[bool] = deque(maxlen=max(1, int(adaptive_lr_window)))
    if use_adaptive_lr:
        initial_lr = float(lr_start) if lr_start is not None else adaptive_lr_floor
        initial_lr = min(max(initial_lr, adaptive_lr_floor), adaptive_lr_max)
        for group in opt.param_groups:
            group["lr"] = initial_lr
        current_lr = initial_lr
    if progress and steps > 0 and progress_step_offset == 0:
        initial_lr = current_lr if use_adaptive_lr else float(lr_start) if use_lr_warmup else base_lrs[0] if base_lrs else None
        _write_progress(progress_label, progress_step_offset, progress_total_steps, 0.0, started, lr=initial_lr)
    for step_idx in range(1, int(steps) + 1):
        if use_lr_warmup:
            frac = min(1.0, (lr_warmup_step_offset + step_idx) / float(lr_warmup_steps))
            for group, base_lr in zip(opt.param_groups, base_lrs):
                group["lr"] = float(lr_start) + frac * (base_lr - float(lr_start))
            current_lr = float(opt.param_groups[0]["lr"]) if opt.param_groups else None
        x, y = data.sample_batch()
        x, y = x.to(device), y.to(device)
        logits, _ = model(x)
        if last_only:
            valid = y[-1] >= 0
            N = int(valid.sum().item())
            if N == 0:
                continue
            task_loss = criterion(logits[-1], y[-1])
        else:
            flat_targets = y.view(-1)
            valid = flat_targets >= 0
            N = int(valid.sum().item())
            if N == 0:
                continue
            task_loss = criterion(logits.view(-1, logits.size(-1)), flat_targets)
        loss = task_loss
        if recurrent_l2_lambda > 0.0 and hasattr(model, "hidden_layer"):
            rec_weight = getattr(model.hidden_layer, "weight", None)
            if rec_weight is not None:
                loss = loss + float(recurrent_l2_lambda) * rec_weight.pow(2).sum()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        opt.step()
        enforce_constraints(model)
        total_loss += task_loss.item() * N
        total_count += N
        recent_loss += task_loss.item() * N
        recent_count += N
        if use_adaptive_lr:
            loss_value = float(task_loss.item())
            if smoothed_loss is None:
                smoothed_loss = loss_value
                best_smoothed_loss = loss_value
            else:
                smoothing = min(1.0, max(0.0, float(adaptive_lr_smoothing)))
                smoothed_loss = (1.0 - smoothing) * smoothed_loss + smoothing * loss_value
                assert best_smoothed_loss is not None
                if smoothed_loss < best_smoothed_loss - float(adaptive_lr_min_delta):
                    best_smoothed_loss = smoothed_loss
                    improvement_streak += 1
                    recent_improvements.append(True)
                    enough_streak = improvement_streak >= int(adaptive_lr_patience)
                    enough_window = (
                        len(recent_improvements) == recent_improvements.maxlen
                        and (sum(recent_improvements) / float(len(recent_improvements))) >= float(adaptive_lr_improve_fraction)
                    )
                    if enough_streak or enough_window:
                        current_lr = min(adaptive_lr_max, float(opt.param_groups[0]["lr"]) * float(adaptive_lr_increase_factor))
                        for group in opt.param_groups:
                            group["lr"] = current_lr
                        improvement_streak = 0
                        recent_improvements.clear()
                elif smoothed_loss > best_smoothed_loss + float(adaptive_lr_min_delta):
                    current_lr = max(adaptive_lr_floor, float(opt.param_groups[0]["lr"]) * float(adaptive_lr_decrease_factor))
                    for group in opt.param_groups:
                        group["lr"] = current_lr
                    improvement_streak = 0
                    recent_improvements.append(False)
                    best_smoothed_loss = smoothed_loss
                else:
                    improvement_streak = 0
                    recent_improvements.append(False)
        if progress and (step_idx % progress_every == 0 or step_idx == int(steps)):
            window_loss = recent_loss / max(1, recent_count)
            _write_progress(
                progress_label,
                progress_step_offset + step_idx,
                progress_total_steps,
                window_loss,
                started,
                lr=current_lr if current_lr is not None else (float(opt.param_groups[0]["lr"]) if opt.param_groups else None),
                final=(progress_step_offset + step_idx) >= progress_total_steps,
            )
            recent_loss, recent_count = 0.0, 0
    return total_loss / max(1, total_count)


@torch.no_grad()
def evaluate(
    model,
    data,
    device,
    criterion,
    *,
    steps: int = 20,
    dataset_last_only: bool = True,
    eval_last_only: bool | None = None,
    response_window_k: int | None = None,
) -> dict:
    """
    Evaluate the model, always tracking final-step accuracy to reflect task decisions.

    dataset_last_only: whether the dataset labels are only valid on the final step.
    eval_last_only: when True, both loss and the primary accuracy use only the final step.
    """
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
    total_response_correct = 0
    total_response_count = 0
    total_response_loss = 0.0
    total_response_loss_weight = 0

    for _ in range(steps):
        x, y = data.sample_batch()
        x, y = x.to(device), y.to(device)
        logits, _ = model(x)

        decision_logits = logits[-1]
        decision_targets = y[-1]
        decision_valid = decision_targets >= 0
        decision_N = int(decision_valid.sum().item())
        decision_loss = criterion(decision_logits, decision_targets)

        if eval_last_only:
            loss_val = decision_loss
            loss_weight = decision_N
        else:
            seq_logits = logits.view(-1, logits.size(-1))
            seq_targets = y.view(-1)
            loss_val = criterion(seq_logits, seq_targets)
            loss_weight = int((seq_targets >= 0).sum().item())

        if loss_weight > 0:
            total_loss += loss_val.item() * loss_weight
            total_loss_weight += loss_weight

        decision_pred = decision_logits.argmax(-1)
        if decision_N > 0:
            decision_correct = ((decision_pred == decision_targets) & decision_valid).sum().item()
            total_decision_correct += decision_correct
            total_decision_count += decision_N

        seq_pred = logits.argmax(-1)
        seq_valid = y >= 0
        seq_total = int(seq_valid.sum().item())
        if seq_total > 0:
            seq_correct = ((seq_pred == y) & seq_valid).sum().item()
            total_seq_correct += seq_correct
            total_seq_count += seq_total

        last_logits, last_targets = last_valid_logits_targets(logits, y)
        if last_logits is not None and last_targets is not None:
            last_loss = criterion(last_logits, last_targets)
            last_pred = last_logits.argmax(-1)
            last_count = int(last_targets.numel())
            total_last_valid_loss += last_loss.item() * last_count
            total_last_valid_loss_weight += last_count
            total_last_valid_correct += int((last_pred == last_targets).sum().item())
            total_last_valid_count += last_count

        if response_window_k is not None and response_window_k > 0:
            k = min(response_window_k, logits.size(0))
            if k > 0:
                resp_logits = logits[-k:].reshape(-1, logits.size(-1))
                resp_targets = y[-k:].reshape(-1)
                resp_loss = criterion(resp_logits, resp_targets)
                resp_weight = int((resp_targets >= 0).sum().item())
                if resp_weight > 0:
                    total_response_loss += resp_loss.item() * resp_weight
                    total_response_loss_weight += resp_weight
                    resp_pred = resp_logits.argmax(-1)
                    resp_valid = resp_targets >= 0
                    resp_correct = ((resp_pred == resp_targets) & resp_valid).sum().item()
                    total_response_correct += resp_correct
                    total_response_count += resp_weight

    mean_loss = total_loss / max(1, total_loss_weight)
    decision_acc = total_decision_correct / max(1, total_decision_count)
    sequence_acc = total_seq_correct / max(1, total_seq_count)
    last_valid_acc = total_last_valid_correct / max(1, total_last_valid_count)
    last_valid_loss = total_last_valid_loss / max(1, total_last_valid_loss_weight)
    response_acc = None
    response_loss = None
    if response_window_k is not None and response_window_k > 0:
        response_acc = total_response_correct / max(1, total_response_count)
        response_loss = total_response_loss / max(1, total_response_loss_weight)

    metrics = {
        "loss": mean_loss,
        "acc": decision_acc,
        "acc_sequence": sequence_acc,
        "acc_last_valid": last_valid_acc,
        "loss_last_valid": last_valid_loss,
    }
    if response_acc is not None:
        metrics["acc_response_window"] = response_acc
        metrics["loss_response_window"] = response_loss
    return metrics
