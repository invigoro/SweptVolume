#!/usr/bin/env python3
"""Collect independent runs into results.csv and print a condition-level summary.

Walks the runs directory, reads each run's summary.json (and test_metrics.json if
final testing has been performed), writes one CSV row per run, and prints the
mean and standard deviation of best evaluation RMSE and training time across the
replicate seeds of each condition.

    python3 aggregate.py                          # all experiments -> results.csv
    python3 aggregate.py --experiment gpu_arch    # one experiment only
    python3 aggregate.py --sort rmse              # rank conditions by evaluation RMSE
"""

import argparse
import csv
import json
import pathlib
import sys

import numpy as np

# Config fields that define an experimental *condition*; everything with the same
# values here differs only by replicate seed.
CONDITION_KEYS = ["experiment", "hidden_layers", "neurons", "lr", "batch_size",
                  "train_size", "epochs", "feature_encoding", "tag"]

COLUMNS = (
    ["experiment", "run_name", "run_dir"]
    + CONDITION_KEYS[1:]
    + ["seed", "status", "diverged_epoch", "best_epoch",
       "best_eval_rmse_liters", "best_eval_mse_liters2",
       "train_time_s", "total_time_s", "epochs_completed",
       "n_parameters", "in_dimension", "steps_per_epoch", "optimizer_steps",
       "test_rmse_liters", "test_mse_liters2",
       "device", "gpu_name", "torch", "python", "git_commit"]
)


def collect(runs_root, experiment=None):
    rows = []
    for summary_path in sorted(runs_root.glob("*/*/summary.json")):
        try:
            data = json.loads(summary_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            print(f"warning: skipping {summary_path}: {exc}", file=sys.stderr)
            continue

        cfg, env = data.get("config", {}), data.get("environment", {})
        if experiment and cfg.get("experiment") != experiment:
            continue

        row = {c: "" for c in COLUMNS}
        row.update({
            "run_name": summary_path.parent.name,
            "run_dir": str(summary_path.parent),
            "experiment": cfg.get("experiment", summary_path.parent.parent.name),
        })
        for key in CONDITION_KEYS[1:] + ["seed"]:
            row[key] = cfg.get(key, "")
        for key in ("status", "diverged_epoch", "best_epoch", "best_eval_rmse_liters",
                    "best_eval_mse_liters2", "train_time_s", "total_time_s",
                    "epochs_completed", "n_parameters", "in_dimension",
                    "steps_per_epoch", "optimizer_steps"):
            row[key] = data.get(key, "")
        for key, src in (("device", "device"), ("gpu_name", "gpu_name"),
                         ("torch", "torch"), ("python", "python"),
                         ("git_commit", "git_commit")):
            row[key] = env.get(src, "")

        # Test metrics exist only after final_test.py has been run on the
        # configuration selected using evaluation data.
        test_path = summary_path.parent / "test_metrics.json"
        if test_path.exists():
            test = json.loads(test_path.read_text())
            row["test_rmse_liters"] = test.get("test_rmse_liters", "")
            row["test_mse_liters2"] = test.get("test_mse_liters2", "")

        rows.append(row)
    return rows


def condition_key(row):
    return tuple(row[k] for k in CONDITION_KEYS)


def summarize(rows, sort_by, expected_seeds):
    """Group rows by condition and reduce over replicate seeds."""
    groups = {}
    for row in rows:
        groups.setdefault(condition_key(row), []).append(row)

    summaries = []
    for key, members in groups.items():
        ok = [m for m in members if m["status"] == "completed"]
        rmse = np.array([float(m["best_eval_rmse_liters"]) for m in ok]) if ok else np.array([])
        ttime = np.array([float(m["train_time_s"]) for m in ok]) if ok else np.array([])
        epochs = np.array([float(m["best_epoch"]) for m in ok]) if ok else np.array([])
        test = np.array([float(m["test_rmse_liters"]) for m in members
                         if m["test_rmse_liters"] != ""])
        summaries.append({
            "key": dict(zip(CONDITION_KEYS, key)),
            "n_seeds": len(members),
            "n_ok": len(ok),
            "seeds": sorted(int(m["seed"]) for m in members),
            "rmse_mean": rmse.mean() if rmse.size else float("nan"),
            "rmse_std": rmse.std(ddof=1) if rmse.size > 1 else 0.0,
            "time_mean": ttime.mean() if ttime.size else float("nan"),
            "time_std": ttime.std(ddof=1) if ttime.size > 1 else 0.0,
            "epoch_mean": epochs.mean() if epochs.size else float("nan"),
            "params": ok[0]["n_parameters"] if ok else "",
            "test_rmse_mean": test.mean() if test.size else None,
            "test_rmse_std": test.std(ddof=1) if test.size > 1 else (0.0 if test.size else None),
            "bad": [m for m in members if m["status"] != "completed"],
        })

    def natural(value):
        """Sort numerically when a config value is a number, textually otherwise."""
        try:
            return (0, float(value), "")
        except (TypeError, ValueError):
            return (1, 0.0, str(value))

    keyfn = {
        "rmse": lambda s: (np.isnan(s["rmse_mean"]), s["rmse_mean"]),
        "time": lambda s: (np.isnan(s["time_mean"]), s["time_mean"]),
        "config": lambda s: tuple(natural(v) for v in s["key"].values()),
    }[sort_by]
    summaries.sort(key=keyfn)

    incomplete = [s for s in summaries if sorted(s["seeds"]) != sorted(expected_seeds)]
    return summaries, incomplete


def print_summary(summaries, incomplete, expected_seeds):
    if not summaries:
        print("no runs found.")
        return

    varying = [k for k in CONDITION_KEYS
               if len({str(s["key"][k]) for s in summaries}) > 1] or ["experiment"]
    width = {k: max(len(k), max(len(str(s["key"][k])) for s in summaries)) for k in varying}

    header = "  ".join(f"{k:>{width[k]}}" for k in varying)
    print(f"\n{header}  {'seeds':>5} {'params':>8} "
          f"{'eval RMSE (L)':>22} {'best epoch':>10} {'train time (s)':>18}")
    print("-" * (len(header) + 64))
    for s in summaries:
        cells = "  ".join(f"{str(s['key'][k]):>{width[k]}}" for k in varying)
        rmse = (f"{s['rmse_mean']:9.4f} +/- {s['rmse_std']:<7.4f}"
                if not np.isnan(s["rmse_mean"]) else f"{'n/a':>21}")
        time_s = (f"{s['time_mean']:7.2f} +/- {s['time_std']:<6.2f}"
                  if not np.isnan(s["time_mean"]) else f"{'n/a':>17}")
        flag = "" if s["n_ok"] == s["n_seeds"] else f"  <- {s['n_seeds'] - s['n_ok']} not completed"
        print(f"{cells}  {s['n_ok']:>5} {str(s['params']):>8} "
              f"{rmse:>22} {s['epoch_mean']:10.1f} {time_s:>18}{flag}")

    with_test = [s for s in summaries if s["test_rmse_mean"] is not None]
    if with_test:
        print("\nfinal held-out test results:")
        for s in with_test:
            cells = ", ".join(f"{k}={s['key'][k]}" for k in varying)
            print(f"  {cells}: test RMSE {s['test_rmse_mean']:.4f} "
                  f"+/- {s['test_rmse_std']:.4f} L")

    if incomplete:
        print(f"\nconditions without all of seeds {expected_seeds}:")
        for s in incomplete:
            cells = ", ".join(f"{k}={s['key'][k]}" for k in varying)
            missing = sorted(set(expected_seeds) - set(s["seeds"]))
            print(f"  {cells}: have {s['seeds']}, missing {missing}")

    bad = [m for s in summaries for m in s["bad"]]
    if bad:
        print("\nruns that did not complete:")
        for m in bad:
            print(f"  {m['run_name']}: {m['status']} at epoch {m['diverged_epoch']}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=pathlib.Path, default=pathlib.Path("runs"),
                    help="root runs directory (default: runs)")
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("results.csv"),
                    help="aggregated CSV path (default: results.csv)")
    ap.add_argument("--experiment", default=None, help="restrict to one experiment")
    ap.add_argument("--seeds", default=None,
                    help="replicate seeds a complete condition should have "
                         "(default: 0,1,2,3,4)")
    ap.add_argument("--sort", choices=["config", "rmse", "time"], default="config",
                    help="ordering of the printed summary (default: config)")
    ap.add_argument("--no-write", action="store_true", help="print the summary without writing CSV")
    args = ap.parse_args(argv)

    expected = ([int(s) for s in args.seeds.split(",")] if args.seeds else [0, 1, 2, 3, 4])

    if not args.runs.is_dir():
        print(f"no runs directory at {args.runs}", file=sys.stderr)
        return 1

    rows = collect(args.runs, args.experiment)
    if not rows:
        print(f"no runs found under {args.runs}"
              + (f" for experiment {args.experiment!r}" if args.experiment else ""))
        return 1

    if not args.no_write:
        with args.out.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=COLUMNS)
            writer.writeheader()
            writer.writerows(sorted(rows, key=lambda r: (r["run_dir"])))
        print(f"wrote {args.out}: {len(rows)} runs")

    summaries, incomplete = summarize(rows, args.sort, expected)
    print_summary(summaries, incomplete, expected)
    return 0


if __name__ == "__main__":
    sys.exit(main())
