"""Command line entry point.

Runs on a real LOBSTER ticker-day when you supply one, and on a generated
synthetic day otherwise, so the repository does something useful immediately
after cloning. Both paths go through identical loading, feature and validation
code -- the synthetic mode writes LOBSTER-format files and reads them back
rather than shortcutting into memory.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from .features import FEATURE_COLUMNS, build_features
from .lobster import SampleSpec, build_snapshots
from .models import FoldResult, build_models, fit_and_score
from .synthetic import generate_lobster_day
from .target import add_target
from .validation import describe_folds, walk_forward_folds


def build_panel(spec: SampleSpec, step_seconds: float, horizon: int) -> pd.DataFrame:
    snapshots = build_snapshots(spec, step_seconds=step_seconds)
    features = build_features(snapshots, levels=spec.levels)
    return add_target(features, horizon)


def run_experiment(
    panel: pd.DataFrame,
    *,
    horizon: int,
    n_folds: int,
    embargo: int,
    min_train: int,
    seed: int = 0,
) -> tuple[list[FoldResult], str]:
    target_column = f"fwdRet_{horizon}"
    columns = [*FEATURE_COLUMNS, target_column]
    clean = panel[columns].replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)

    print(f"panel rows: {len(panel)}  |  complete rows after feature warm-up: {len(clean)}")
    if len(clean) < min_train * 2:
        raise SystemExit(
            f"Only {len(clean)} complete rows; need at least {min_train * 2}. "
            "Use a smaller --step-seconds, a shorter --horizon, or a lower --min-train."
        )

    x = clean[list(FEATURE_COLUMNS)].to_numpy(dtype=float)
    y = clean[target_column].to_numpy(dtype=float)

    folds = walk_forward_folds(
        len(clean), n_folds=n_folds, horizon=horizon, embargo=embargo, min_train=min_train
    )
    results: list[FoldResult] = []
    for index, fold in enumerate(folds, start=1):
        for name, model in build_models(seed).items():
            results.append(
                fit_and_score(
                    name,
                    model,
                    x[fold.train],
                    y[fold.train],
                    x[fold.test],
                    y[fold.test],
                    index,
                    list(FEATURE_COLUMNS),
                )
            )
    return results, describe_folds(folds)


def summarize(results: list[FoldResult]) -> pd.DataFrame:
    frame = pd.DataFrame(
        [
            {
                "fold": r.fold,
                "model": r.model,
                "nTrain": r.n_train,
                "nTest": r.n_test,
                "r2InSample": r.r2_in,
                "r2OutOfSample": r.r2_out,
                "hitRate": r.hit_rate,
            }
            for r in results
        ]
    )
    return frame


def print_summary(frame: pd.DataFrame) -> None:
    print("\n=== per fold ===")
    print(frame.round(5).to_string(index=False))

    print("\n=== mean across folds ===")
    grouped = frame.groupby("model")[["r2InSample", "r2OutOfSample", "hitRate"]].mean()
    print(grouped.round(5).to_string())

    print(
        "\nOut-of-sample R2 is measured against the TRAINING mean. Negative values\n"
        "mean the model does worse than predicting that constant, which is the\n"
        "normal outcome for short-horizon return prediction and is reported as-is."
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="obrp",
        description="Short-horizon order book return prediction with purged walk-forward validation.",
    )
    parser.add_argument("--message-file", type=Path, help="LOBSTER message CSV.")
    parser.add_argument("--orderbook-file", type=Path, help="LOBSTER orderbook CSV.")
    parser.add_argument("--levels", type=int, default=10, help="Book levels in the files. Default: 10.")
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Generate a synthetic LOBSTER-format day instead of reading real files.",
    )
    parser.add_argument(
        "--synthetic-effect",
        type=float,
        default=0.35,
        help="Planted imbalance-to-return effect in synthetic mode. 0 gives a pure random walk.",
    )
    parser.add_argument("--step-seconds", type=float, default=1.0, help="Grid spacing. Default: 1.0.")
    parser.add_argument("--horizon", type=int, default=10, help="Forward steps to predict. Default: 10.")
    parser.add_argument("--n-folds", type=int, default=5, help="Walk-forward folds. Default: 5.")
    parser.add_argument(
        "--embargo",
        type=int,
        default=50,
        help="Steps dropped from the start of each test period. Default: 50.",
    )
    parser.add_argument("--min-train", type=int, default=2000, help="Minimum training rows. Default: 2000.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.synthetic:
        with tempfile.TemporaryDirectory() as tmp:
            spec = generate_lobster_day(
                Path(tmp), imbalance_effect=args.synthetic_effect, seed=args.seed
            )
            print(f"synthetic day written to {spec.message_path.name}")
            panel = build_panel(spec, args.step_seconds, args.horizon)
    else:
        if not args.message_file or not args.orderbook_file:
            raise SystemExit(
                "Supply --message-file and --orderbook-file, or pass --synthetic. "
                "See README for the LOBSTER sample download."
            )
        spec = SampleSpec(
            message_path=args.message_file,
            orderbook_path=args.orderbook_file,
            levels=args.levels,
            ticker=args.message_file.stem.split("_")[0],
        )
        panel = build_panel(spec, args.step_seconds, args.horizon)

    results, fold_table = run_experiment(
        panel,
        horizon=args.horizon,
        n_folds=args.n_folds,
        embargo=args.embargo,
        min_train=args.min_train,
        seed=args.seed,
    )

    print("\n=== folds (after purge and embargo) ===")
    print(fold_table)

    frame = summarize(results)
    print_summary(frame)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / "fold_results.csv"
    frame.to_csv(path, index=False)
    print(f"\nwrote {path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
