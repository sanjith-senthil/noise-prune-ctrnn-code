import argparse
import warnings
from typing import Tuple

from .analysis.summary import summarize_csv
from .experiments import (
    run_prune_experiment,
    run_suite_from_config,
    train_baselines,
)
from .pruning import available_pruning_strategies
from .utils import make_run_id, set_global_seed

warnings.filterwarnings("ignore", message=".*Gym has been unmaintained.*")
warnings.filterwarnings("ignore", message=".*migration_guide.*")
warnings.filterwarnings("ignore", message=".*env\\.gt.*", category=UserWarning)


def _parse_comma_floats(src: str) -> Tuple[float, ...]:
    return tuple(float(item) for item in src.split(",") if item.strip())


def _parse_comma_ints(src: str) -> Tuple[int, ...]:
    return tuple(int(item) for item in src.split(",") if item.strip())


def _parse_comma_strs(src: str) -> Tuple[str, ...]:
    return tuple(item.strip() for item in src.split(",") if item.strip())


def main():
    parser = argparse.ArgumentParser()
    pruning_choices = ["none"] + sorted(available_pruning_strategies().keys())
    parser.add_argument("--strategy", default="l1_unstructured", choices=pruning_choices)
    parser.add_argument("--amount", type=float, default=0.5)
    parser.add_argument("--train_steps", type=int, default=600)
    parser.add_argument("--ft_steps", type=int, default=200)

    last_only_group = parser.add_mutually_exclusive_group()
    last_only_group.add_argument(
        "--last_only",
        dest="last_only",
        action="store_true",
        help="Evaluate/train using only the final timestep (default).",
    )
    last_only_group.add_argument(
        "--full_sequence",
        dest="last_only",
        action="store_false",
        help="Evaluate loss/accuracy across the entire sequence.",
    )
    parser.set_defaults(last_only=True)

    eval_last_only_group = parser.add_mutually_exclusive_group()
    eval_last_only_group.add_argument(
        "--eval_last_only",
        dest="eval_last_only",
        action="store_true",
        help="Evaluate accuracy using only the final timestep (default).",
    )
    eval_last_only_group.add_argument(
        "--eval_full_sequence",
        dest="eval_last_only",
        action="store_false",
        help="Evaluate loss/accuracy across the entire sequence.",
    )
    parser.set_defaults(eval_last_only=True)

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--movement_batches",
        type=int,
        default=20,
        help="Number of calibration batches used by pruning strategies that require data.",
    )
    parser.add_argument(
        "--mode",
        choices=["single", "suite", "summary", "baseline"],
        default="single",
    )
    parser.add_argument("--out_csv", default=None)
    parser.add_argument(
        "--task",
        default="modcog:ctxdlydm1seql",
        help="modcog:<Mod-Cog task>",
    )
    parser.add_argument("--no_prune", action="store_true", help="Skip pruning (control)")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a suite configuration file (required for mode=suite)",
    )
    parser.add_argument(
        "--input_csv",
        type=str,
        default=None,
        help="Input CSV for summary mode",
    )
    parser.add_argument(
        "--summary_out",
        type=str,
        default=None,
        help="Optional output path (.csv or .json) for summary mode",
    )
    parser.add_argument(
        "--group_by",
        type=str,
        default="strategy,amount",
        help="Comma-separated columns to group by when summarising",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        default="post_acc_sequence,post_loss",
        help="Comma-separated metric columns to average",
    )
    parser.add_argument(
        "--ng_kwargs",
        type=str,
        default=None,
        help='JSON dict passed to Mod-Cog task constructors, e.g. "{\\"sigma\\": 1.0}"',
    )
    parser.add_argument(
        "--ng_dataset_kwargs",
        type=str,
        default=None,
        help="Legacy suite-schema field retained for compatibility; ignored by trial-aligned Mod-Cog sampling.",
    )
    parser.add_argument("--ng_T", type=int, default=None, help="Override Mod-Cog trial length T")
    parser.add_argument("--ng_B", type=int, default=None, help="Override Mod-Cog batch size B")
    parser.add_argument("--hidden_size", type=int, default=None, help="Override RNN hidden size")
    parser.add_argument(
        "--activation",
        choices=["tanh", "shifted_tanh", "relu", "softplus"],
        default="tanh",
        help="CTRNN activation function.",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["ctrnn"],
        default="ctrnn",
        help="Base model architecture. The public release supports CTRNN only.",
    )
    parser.add_argument("--noise_sigma", type=float, default=1.0, help="Noise prune sigma hyperparameter")
    parser.add_argument("--noise_eps", type=float, default=0.3, help="Noise prune epsilon hyperparameter")
    parser.add_argument(
        "--noise_leak_shift",
        type=float,
        default=0.0,
        help="Shift applied to the CT operator diagonal during noise pruning",
    )
    parser.add_argument(
        "--noise_matched_diagonal",
        type=int,
        choices=[0, 1],
        default=1,
        help="Set to 0 to disable matched diagonal in noise pruning",
    )
    parser.add_argument("--noise_rng_seed", type=int, default=None, help="Optional RNG seed for noise pruning")
    parser.add_argument("--sim_np_sigma", type=float, default=None)
    parser.add_argument("--sim_np_sigma_source", type=str, default="natural_voltage")
    parser.add_argument("--sim_np_observable_space", type=str, default="rate")
    parser.add_argument("--sim_np_inject_space", type=str, default="rate")
    parser.add_argument("--sim_np_centering", type=str, default="trajectory_mean")
    parser.add_argument("--sim_np_max_samples", type=int, default=25000)
    parser.add_argument("--sim_np_burn_in_steps", type=int, default=300)
    parser.add_argument(
        "--rescale_cap_mode",
        choices=["fixed", "quantile"],
        default="quantile",
        help="Cap mode for capped-rescale noise-prune strategies.",
    )
    parser.add_argument("--rescale_cap_value", type=float, default=None)
    parser.add_argument("--rescale_cap_quantile", type=float, default=0.95)
    parser.add_argument("--obs_damping", type=float, default=1e-3)
    parser.add_argument("--obs_max_samples", type=int, default=25000)
    parser.add_argument(
        "--obs_compensation_mode",
        choices=["diagonal", "block", "auto"],
        default="diagonal",
        help="Compensated OBS update mode for strategy=obs_compensated.",
    )
    parser.add_argument("--obs_exact_block_threshold", type=int, default=128)
    parser.add_argument("--eval_seed", type=int, default=None, help="Seed for deterministic evaluation sampling")
    parser.add_argument(
        "--eval_sample_batches",
        type=int,
        default=32,
        help="Number of fixed batches to reuse during evaluation (set 0 to disable deterministic eval)",
    )
    parser.add_argument("--eval_steps_pre0", type=int, default=50)
    parser.add_argument("--eval_steps_pre", type=int, default=100)
    parser.add_argument("--eval_steps_post0", type=int, default=100)
    parser.add_argument("--eval_steps_post", type=int, default=100)
    parser.add_argument(
        "--score_batch_seed",
        type=int,
        default=None,
        help="Optional seed for pruning-score batches, independent of train and eval seeds.",
    )
    parser.add_argument(
        "--recurrent_l2_lambda",
        type=float,
        default=0.0,
        help="Optional Frobenius penalty coefficient for CTRNN recurrent weights.",
    )
    parser.add_argument(
        "--train_select_best",
        action="store_true",
        help="During training, periodically validate and restore the best checkpoint before saving.",
    )
    parser.add_argument(
        "--train_val_interval",
        type=int,
        default=0,
        help="Training steps between validation checkpoints when --train_select_best is enabled.",
    )
    parser.add_argument("--train_best_metric", default="acc_sequence")
    parser.add_argument("--train_best_metric_mode", choices=["max", "min"], default="max")
    parser.add_argument("--train_best_tie_metric", default="loss_last_valid")
    parser.add_argument("--train_best_tie_metric_mode", choices=["max", "min"], default="min")
    parser.add_argument(
        "--train_history_path",
        default=None,
        help="Optional path for periodic training validation history CSV.",
    )
    parser.add_argument(
        "--skip_training",
        action="store_true",
        help="Legacy flag. For baseline mode, use --overwrite instead.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing baseline checkpoints.",
    )
    parser.add_argument(
        "--save_model_path",
        type=str,
        default=None,
        help="Optional path to save the trained model before pruning",
    )
    parser.add_argument(
        "--load_model_path",
        type=str,
        default=None,
        help="Optional path to load a pre-trained model before running the experiment",
    )
    parser.add_argument(
        "--baseline_config",
        type=str,
        default=None,
        help="Path to a baseline-training configuration file (mode=baseline)",
    )
    parser.add_argument(
        "--run_id",
        type=str,
        default=None,
        help="Optional identifier for this run (defaults to timestamp)",
    )

    args = parser.parse_args()

    set_global_seed(args.seed)
    run_id = args.run_id or make_run_id()
    out_csv = args.out_csv or f"results/{run_id}.csv"

    overwrite_flag = bool(args.overwrite or args.skip_training)

    if args.mode == "baseline":
        if args.baseline_config is None:
            raise ValueError("--baseline_config is required when mode=baseline")
        checkpoints = train_baselines(args.baseline_config, overwrite=overwrite_flag)
        for path in checkpoints:
            print(path)
        return

    if args.mode == "suite":
        if args.config is None:
            raise ValueError("--config is required when mode=suite")
        path = run_suite_from_config(args.config)
        print("Suite wrote:", path)
        return

    if args.mode == "summary":
        if args.input_csv is None:
            raise ValueError("--input_csv is required when mode=summary")
        group_fields = _parse_comma_strs(args.group_by) or ("strategy", "amount")
        metric_fields = _parse_comma_strs(args.metrics) or ("post_acc", "post_loss")
        summaries = summarize_csv(
            args.input_csv,
            group_fields=group_fields,
            metrics=metric_fields,
            output_path=args.summary_out,
        )
        for row in summaries:
            print(row)
        return

    res = run_prune_experiment(
        strategy=args.strategy,
        amount=args.amount,
        train_steps=args.train_steps,
        ft_steps=args.ft_steps,
        last_only=args.last_only,
        model_type=args.model_type,
        eval_last_only=args.eval_last_only,
        seed=args.seed,
        device=args.device,
        movement_batches=args.movement_batches,
        task=args.task,
        no_prune=args.no_prune,
        ng_kwargs=args.ng_kwargs,
        ng_dataset_kwargs=args.ng_dataset_kwargs,
        ng_T=args.ng_T,
        ng_B=args.ng_B,
        hidden_size=args.hidden_size,
        activation=args.activation,
        noise_sigma=args.noise_sigma,
        noise_eps=args.noise_eps,
        noise_leak_shift=args.noise_leak_shift,
        noise_matched_diagonal=bool(args.noise_matched_diagonal),
        noise_rng_seed=args.noise_rng_seed,
        sim_np_sigma=args.sim_np_sigma,
        sim_np_sigma_source=args.sim_np_sigma_source,
        sim_np_observable_space=args.sim_np_observable_space,
        sim_np_inject_space=args.sim_np_inject_space,
        sim_np_centering=args.sim_np_centering,
        sim_np_max_samples=args.sim_np_max_samples,
        sim_np_burn_in_steps=args.sim_np_burn_in_steps,
        rescale_cap_mode=args.rescale_cap_mode,
        rescale_cap_value=args.rescale_cap_value,
        rescale_cap_quantile=args.rescale_cap_quantile,
        obs_damping=args.obs_damping,
        obs_max_samples=args.obs_max_samples,
        obs_compensation_mode=args.obs_compensation_mode,
        obs_exact_block_threshold=args.obs_exact_block_threshold,
        eval_seed=args.eval_seed,
        eval_sample_batches=args.eval_sample_batches,
        eval_steps_pre0=args.eval_steps_pre0,
        eval_steps_pre=args.eval_steps_pre,
        eval_steps_post0=args.eval_steps_post0,
        eval_steps_post=args.eval_steps_post,
        score_batch_seed=args.score_batch_seed,
        recurrent_l2_lambda=args.recurrent_l2_lambda,
        train_select_best=args.train_select_best,
        train_val_interval=args.train_val_interval,
        train_best_metric=args.train_best_metric,
        train_best_metric_mode=args.train_best_metric_mode,
        train_best_tie_metric=args.train_best_tie_metric,
        train_best_tie_metric_mode=args.train_best_tie_metric_mode,
        train_history_path=args.train_history_path,
        skip_training=args.skip_training,
        save_model_path=args.save_model_path,
        load_model_path=args.load_model_path,
        run_id=run_id,
    )
    print(res)


if __name__ == "__main__":
    main()
