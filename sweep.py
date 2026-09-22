#!/usr/bin/env python3
"""Drive many independent runs of neural_network.py and combine them into a sweep.

Every run remains independent: the same run can equally be launched by hand with
neural_network.py, and runs may be split across machines with --shard. This
script only enumerates the grid, skips runs that are already complete, and
reports progress.

    python3 sweep.py --sweep gpu_arch                  # the required GPU sweep
    python3 sweep.py --sweep gpu_arch --dry-run        # list what would run
    python3 sweep.py --sweep gpu_arch --shard 1/3      # this machine's third

Runs execute in-process by default. These networks train in a few seconds, so a
fresh `import torch` per run (~2 s) would dominate both the wall clock and the
timing comparison; --subprocess is available when full isolation is wanted.
"""

import argparse
import itertools
import pathlib
import subprocess
import sys
import time

import neural_network as nn_mod

# Defaults shared by every condition unless a sweep or --set overrides them.
BASE = dict(
    experiment="gpu_arch", hidden_layers=2, neurons=128, lr=1e-3, batch_size=10000,
    train_size=100000, epochs=1000, seed=0, device="auto", feature_encoding="raw",
    track_train_metric=True, data_file=None, data_dir=None, tag=None,
)

# Named sweeps. 'grid' maps a config key to the list of values it takes; the
# sweep is the Cartesian product of those lists crossed with the replicate seeds.
SWEEPS = {
    # The required GPU experiment: network depth and width at fixed lr, batch
    # size, and training-data size. 3 x 4 x 5 seeds = 60 runs.
    "gpu_arch": {
        "experiment": "gpu_arch",
        "grid": {"hidden_layers": [1, 2, 3], "neurons": [32, 64, 128, 256]},
        "overrides": {},
    },
    # The CPU-track grid. Not required for the GPU assignment, but supported
    # because the implementation handles training-data subsets, and it is a
    # legitimate basis for a self-directed comparison.
    "cpu_lr_size": {
        "experiment": "cpu_lr_size",
        "grid": {"lr": [1e-4, 1e-3, 1e-2, 1e-1], "train_size": [1000, 10000, 100000]},
        "overrides": {"hidden_layers": 1, "neurons": 64, "batch_size": 1000},
    },
}

INT_KEYS = {"hidden_layers", "neurons", "batch_size", "train_size", "epochs", "seed"}
FLOAT_KEYS = {"lr"}
BOOL_KEYS = {"track_train_metric"}


def coerce(key, value):
    """Turn a command-line string into the type the config expects."""
    if key in INT_KEYS:
        return int(value)
    if key in FLOAT_KEYS:
        return float(value)
    if key in BOOL_KEYS:
        return str(value).lower() in ("1", "true", "yes", "on")
    return value


def parse_assignments(items, allow_lists):
    """Parse `key=value` or `key=v1,v2,v3` arguments into a dict."""
    out = {}
    for item in items or []:
        if "=" not in item:
            raise SystemExit(f"expected key=value, got {item!r}")
        key, _, raw = item.partition("=")
        key = key.strip().replace("-", "_")
        if key not in BASE:
            raise SystemExit(f"unknown configuration key {key!r}; valid keys: "
                             + ", ".join(sorted(BASE)))
        values = [coerce(key, v) for v in raw.split(",")] if allow_lists else coerce(key, raw)
        out[key] = values
    return out


def build_configs(args):
    """Expand the requested sweep into an ordered list of run configurations."""
    cfg_base = dict(BASE)
    grid = {}
    experiment = None

    if args.sweep:
        if args.sweep not in SWEEPS:
            raise SystemExit(f"unknown sweep {args.sweep!r}; known: " + ", ".join(SWEEPS))
        spec = SWEEPS[args.sweep]
        experiment = spec["experiment"]
        cfg_base.update(spec["overrides"])
        grid.update(spec["grid"])

    grid.update(parse_assignments(args.grid, allow_lists=True))
    cfg_base.update(parse_assignments(args.set, allow_lists=False))

    if args.experiment:
        experiment = args.experiment
    if experiment is None:
        raise SystemExit("specify --sweep or --experiment")
    cfg_base["experiment"] = experiment

    for key in ("device", "data_file", "data_dir", "tag"):
        value = getattr(args, key if key != "data_file" else "data")
        if value is not None:
            cfg_base[key] = str(value) if key.startswith("data") else value

    if not grid:
        raise SystemExit("no grid to sweep: pass --sweep or at least one --grid key=v1,v2")

    keys = list(grid)
    configs = []
    # Seeds vary fastest so that all five replicates of a condition finish together.
    for combo in itertools.product(*(grid[k] for k in keys)):
        for seed in args.seeds:
            cfg = nn_mod.make_config(**{**cfg_base, **dict(zip(keys, combo)), "seed": seed})
            configs.append(cfg)
    return configs


def run_subprocess(cfg, out_root, replace):
    """Launch one run as a separate process, for full isolation."""
    cmd = [sys.executable, "neural_network.py",
           "--experiment", cfg["experiment"],
           "--hidden-layers", str(cfg["hidden_layers"]),
           "--neurons", str(cfg["neurons"]),
           "--lr", repr(cfg["lr"]),
           "--batch-size", str(cfg["batch_size"]),
           "--train-size", str(cfg["train_size"]),
           "--epochs", str(cfg["epochs"]),
           "--seed", str(cfg["seed"]),
           "--device", cfg["device"],
           "--feature-encoding", cfg["feature_encoding"],
           "--out", str(out_root), "--quiet"]
    if cfg["seed"] not in nn_mod.REPLICATE_SEEDS:
        cmd.append("--any-seed")
    if cfg.get("tag"):
        cmd += ["--tag", cfg["tag"]]
    if cfg.get("data_file"):
        cmd += ["--data", cfg["data_file"]]
    if cfg.get("data_dir"):
        cmd += ["--data-dir", cfg["data_dir"]]
    if not cfg["track_train_metric"]:
        cmd.append("--no-train-metric")
    if replace:
        cmd.append("--replace")
    return subprocess.run(cmd).returncode


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweep", help="named sweep: " + ", ".join(SWEEPS))
    ap.add_argument("--experiment", help="experiment name (overrides the sweep's own)")
    ap.add_argument("--grid", action="append", metavar="KEY=V1,V2",
                    help="add or replace a swept dimension; repeatable")
    ap.add_argument("--set", action="append", metavar="KEY=VALUE",
                    help="fix a configuration value for every run; repeatable")
    ap.add_argument("--seeds", default=",".join(map(str, nn_mod.REPLICATE_SEEDS)),
                    help="comma-separated replicate seeds (default: 0,1,2,3,4)")
    ap.add_argument("--shard", metavar="I/N",
                    help="run only shard I of N (1-based), to split across machines")
    ap.add_argument("--device", default=None, help="auto, cuda, cuda:N, or cpu")
    ap.add_argument("--data", type=pathlib.Path, default=None, help="dataset .npz path")
    ap.add_argument("--data-dir", type=pathlib.Path, default=None, dest="data_dir",
                    help="fallback directory of .npy arrays")
    ap.add_argument("--tag", default=None, help="label folded into every run directory name")
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("runs"),
                    help="root output directory (default: runs)")
    ap.add_argument("--replace", action="store_true", help="re-run and overwrite completed runs")
    ap.add_argument("--dry-run", action="store_true", help="list the runs and exit")
    ap.add_argument("--subprocess", action="store_true",
                    help="run each configuration in a separate process")
    ap.add_argument("--stop-on-error", action="store_true",
                    help="abort the sweep if a run diverges or raises (default: continue)")
    args = ap.parse_args(argv)

    args.seeds = [int(s) for s in args.seeds.split(",") if s != ""]
    configs = build_configs(args)

    if args.shard:
        try:
            index, total = (int(x) for x in args.shard.split("/"))
        except ValueError:
            raise SystemExit("--shard must look like 1/3")
        if not 1 <= index <= total:
            raise SystemExit(f"--shard index must be in 1..{total}")
        configs = configs[index - 1::total]
        print(f"shard {index}/{total}: {len(configs)} of the full grid")

    pending, done = [], []
    for cfg in configs:
        out_dir = args.out / cfg["experiment"] / nn_mod.run_name(cfg)
        (done if (out_dir / "summary.json").exists() and not args.replace
         else pending).append((cfg, out_dir))

    print(f"{len(configs)} runs requested, {len(done)} already complete, {len(pending)} to run")
    if args.dry_run:
        for cfg, out_dir in pending:
            print(f"  TODO {out_dir}")
        for _, out_dir in done:
            print(f"  done {out_dir}")
        return 0
    if not pending:
        print("nothing to do.")
        return 0

    started = time.perf_counter()
    failures = []
    for i, (cfg, out_dir) in enumerate(pending, start=1):
        elapsed = time.perf_counter() - started
        eta = f", eta {elapsed / (i - 1) * (len(pending) - i + 1):6.0f}s" if i > 1 else ""
        print(f"[{i:>3}/{len(pending)}] {out_dir.name}{eta}", flush=True)

        try:
            if args.subprocess:
                code = run_subprocess(cfg, args.out, args.replace)
                if code != 0:
                    failures.append((out_dir.name, f"exit code {code}"))
                continue
            result = nn_mod.train_one_run(cfg, verbose=False)
            nn_mod.save_run(out_dir, cfg, result)
            s = result["summary"]
            if s["status"] != "completed":
                failures.append((out_dir.name, s["status"]))
                print(f"        {s['status']} at epoch {s['diverged_epoch']}")
            else:
                print(f"        best eval RMSE {s['best_eval_rmse_liters']:8.4f} L "
                      f"@ epoch {s['best_epoch']:<5} train {s['train_time_s']:6.2f} s")
        except Exception as exc:                        # keep the sweep going by default
            failures.append((out_dir.name, f"{type(exc).__name__}: {exc}"))
            print(f"        ERROR {type(exc).__name__}: {exc}")
            if args.stop_on_error:
                raise

    total = time.perf_counter() - started
    print(f"\n{len(pending) - len(failures)}/{len(pending)} runs completed in {total:.1f} s")
    if failures:
        print("runs needing attention (recorded, not silently retried):")
        for name, why in failures:
            print(f"  {name}: {why}")
    print(f"\naggregate with:  python3 aggregate.py --runs {args.out}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
