#!/usr/bin/env python3
"""Evaluate the five retained models of ONE selected configuration on the held-out test set.

This is the only code in the project that reads the testing arrays. The
assignment requires the final configuration to be chosen using evaluation
results alone, so this script is deliberately awkward to run by accident: it
names the configuration explicitly, refuses to run on more than one
configuration at a time, and requires --confirm.

    # look at what would be tested, without touching the testing data
    python3 final_test.py --experiment gpu_arch --hidden-layers 3 --neurons 256

    # actually run it, once, after the configuration has been selected
    python3 final_test.py --experiment gpu_arch --hidden-layers 3 --neurons 256 --confirm

Results are written to test_metrics.json inside each run directory and picked up
by aggregate.py, so results.csv carries the final test numbers alongside
everything else.
"""

import argparse
import json
import pathlib
import sys

import numpy as np
import torch

import neural_network as nn_mod


def find_runs(runs_root, experiment, hidden_layers, neurons, extra):
    """Locate the run directories of one configuration, one per replicate seed."""
    base = runs_root / experiment
    if not base.is_dir():
        raise SystemExit(f"no experiment directory at {base}")

    matches = []
    for summary_path in sorted(base.glob("*/summary.json")):
        data = json.loads(summary_path.read_text())
        cfg = data.get("config", {})
        if cfg.get("hidden_layers") != hidden_layers or cfg.get("neurons") != neurons:
            continue
        if any(cfg.get(k) != v for k, v in extra.items()):
            continue
        matches.append((summary_path.parent, data))
    return sorted(matches, key=lambda m: m[1]["config"]["seed"])


def evaluate(run_dir, data, device):
    """Score one retained checkpoint on the testing set, in liters."""
    checkpoint = torch.load(run_dir / "best_model.pt", map_location=device, weights_only=False)
    cfg, norm = checkpoint["config"], checkpoint["normalization"]

    x = torch.tensor(data["testing_features"], dtype=torch.float32, device=device)
    y = torch.tensor(data["testing_labels"], dtype=torch.float32, device=device)
    x = nn_mod.encode_features(x, cfg["feature_encoding"])

    # The saved normalization statistics are the ones derived from that run's
    # training data; nothing is recomputed from the testing set.
    to = lambda a: torch.tensor(a, dtype=torch.float32, device=device)
    x = (x - to(norm["x_mean"])) / to(norm["x_std"])

    model = nn_mod.NeuralNetwork(
        in_dimension=x.shape[1], out_dimension=y.shape[1],
        hidden_layers=cfg["hidden_layers"],
        neurons_per_hidden_layer=cfg["neurons"]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    with torch.no_grad():
        predicted = model(x) * to(norm["y_std"]) + to(norm["y_mean"])
        mse = float(((predicted - y) ** 2).mean())

    return {
        "seed": cfg["seed"],
        "best_epoch": checkpoint["best_epoch"],
        "eval_rmse_liters": checkpoint["best_eval_rmse_liters"],
        "test_mse_liters2": mse,
        "test_rmse_liters": mse ** 0.5,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiment", default="gpu_arch")
    ap.add_argument("--hidden-layers", type=int, required=True)
    ap.add_argument("--neurons", type=int, required=True)
    ap.add_argument("--feature-encoding", default=None,
                    help="restrict to a feature encoding, if the experiment mixes them")
    ap.add_argument("--epochs", type=int, default=None,
                    help="restrict to an epoch budget, if the experiment mixes them")
    ap.add_argument("--batch-size", type=int, default=None,
                    help="restrict to a batch size, if the experiment mixes them")
    ap.add_argument("--lr", type=float, default=None,
                    help="restrict to a learning rate, if the experiment mixes them")
    ap.add_argument("--train-size", type=int, default=None,
                    help="restrict to a training-data size, if the experiment mixes them")
    ap.add_argument("--runs", type=pathlib.Path, default=pathlib.Path("runs"))
    ap.add_argument("--device", default="auto")
    ap.add_argument("--expect-seeds", type=int, default=5,
                    help="number of retained models expected (default: 5)")
    ap.add_argument("--confirm", action="store_true",
                    help="required: confirms the configuration was selected on evaluation data")
    ap.add_argument("--replace", action="store_true",
                    help="overwrite existing test_metrics.json files")
    args = ap.parse_args(argv)

    extra = {}
    if args.feature_encoding is not None:
        extra["feature_encoding"] = args.feature_encoding
    if args.epochs is not None:
        extra["epochs"] = args.epochs
    if args.batch_size is not None:
        extra["batch_size"] = args.batch_size
    if args.lr is not None:
        extra["lr"] = args.lr
    if args.train_size is not None:
        extra["train_size"] = args.train_size

    matches = find_runs(args.runs, args.experiment, args.hidden_layers, args.neurons, extra)
    if not matches:
        raise SystemExit(f"no runs matching hidden_layers={args.hidden_layers}, "
                         f"neurons={args.neurons} in {args.runs / args.experiment}")

    print(f"configuration: {args.hidden_layers} hidden layers x {args.neurons} neurons"
          + "".join(f", {k}={v}" for k, v in extra.items()))
    print(f"retained models found: {len(matches)}")
    for run_dir, data in matches:
        print(f"  seed {data['config']['seed']}: {run_dir.name}  "
              f"(eval RMSE {data['best_eval_rmse_liters']:.4f} L @ epoch {data['best_epoch']})")

    if len(matches) != args.expect_seeds:
        print(f"\nexpected {args.expect_seeds} retained models but found {len(matches)}.",
              file=sys.stderr)
        seeds = [m[1]["config"]["seed"] for m in matches]
        if len(seeds) != len(set(seeds)):
            # More than one configuration matched, so the selection is ambiguous.
            # Name the fields that differ, so the user knows what to add.
            differing = sorted({k for k in ("lr", "batch_size", "epochs", "train_size",
                                            "feature_encoding", "tag")
                                if len({str(m[1]["config"].get(k)) for m in matches}) > 1})
            print("Duplicate seeds: more than one configuration matched. "
                  "These fields differ between them:", file=sys.stderr)
            for key in differing:
                values = sorted({str(m[1]["config"].get(key)) for m in matches})
                print(f"  {key}: {', '.join(values)}"
                      f"   (narrow with --{key.replace('_', '-')})", file=sys.stderr)
        else:
            print("Pass --expect-seeds to override if this is intentional.", file=sys.stderr)
        return 1

    existing = [d for d, _ in matches if (d / "test_metrics.json").exists()]
    if existing and not args.replace:
        print(f"\n{len(existing)} of these runs already have test_metrics.json.", file=sys.stderr)
        print("Final testing should happen once. Pass --replace to overwrite.", file=sys.stderr)
        return 2

    if not args.confirm:
        print("\nThis reads the held-out testing set, which must happen only after the")
        print("final configuration has been chosen using evaluation results alone.")
        print("Nothing was read. Re-run with --confirm to proceed.")
        return 0

    device = nn_mod.resolve_device(args.device)
    data, source = nn_mod.load_dataset(include_test=True)
    print(f"\nloading testing data from {source} on {device}")

    results = []
    for run_dir, _ in matches:
        result = evaluate(run_dir, data, device)
        (run_dir / "test_metrics.json").write_text(json.dumps(result, indent=2) + "\n")
        results.append(result)
        print(f"  seed {result['seed']}: test RMSE {result['test_rmse_liters']:8.4f} L   "
              f"test MSE {result['test_mse_liters2']:10.4f} L^2")

    rmse = np.array([r["test_rmse_liters"] for r in results])
    mse = np.array([r["test_mse_liters2"] for r in results])
    evl = np.array([r["eval_rmse_liters"] for r in results])

    print(f"\nfinal held-out test results over {len(results)} retained models")
    print(f"  test MSE  : {mse.mean():10.4f} +/- {mse.std(ddof=1):.4f} L^2")
    print(f"  test RMSE : {rmse.mean():10.4f} +/- {rmse.std(ddof=1):.4f} L")
    print(f"  (evaluation RMSE for the same models: "
          f"{evl.mean():.4f} +/- {evl.std(ddof=1):.4f} L)")
    print("\nwritten to test_metrics.json in each run directory; "
          "re-run aggregate.py to fold into results.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
