#!/usr/bin/env python3
"""Fully connected neural network and training program for the swept-volume task.

Predicts the volume of space (in liters) swept by a seven-jointed robot moving
between a starting and an ending joint configuration, from the 14 joint angles.

One invocation trains ONE configuration with ONE replicate seed, so independent
runs can be distributed across machines and combined afterwards. See README.md
for the exact commands; `sweep.py` drives the full set of runs.

    python3 neural_network.py --hidden-layers 2 --neurons 128 --seed 0
    python3 neural_network.py --quick-test
"""

import argparse
import copy
import json
import pathlib
import platform
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn as nn

# The five replicate seeds required by the assignment, used for every condition.
REPLICATE_SEEDS = [0, 1, 2, 3, 4]

# Fixed seed defining the one random ordering of the 100,000 training examples
# used to carve out the 1,000- and 10,000-example subsets. It is deliberately
# NOT the replicate seed: the assignment requires the same subsets across all
# five replicates, and requires the smaller subset to be nested in the larger.
SUBSET_ORDER_SEED = 20250917

DEFAULT_DATA_FILE = pathlib.Path("swept_volume_data.npz")
DEFAULT_DATA_DIR = pathlib.Path("source_data/swept_volume_data")

FEATURE_ARRAYS = ["training", "evaluation", "testing"]


# --------------------------------------------------------------------------- #
# Network
# --------------------------------------------------------------------------- #

class NeuralNetwork(nn.Module):
    """A fully connected network with a variable number of equally wide hidden layers.

    Weights use Xavier uniform initialisation and biases are zeroed, as required
    by the assignment. ReLU is applied after every layer except the last, so the
    output layer is linear and can produce unbounded regression targets.

    With hidden_layers == 0 the network degenerates to a single linear map; this
    is not part of the required sweep but is useful as a baseline.
    """

    def __init__(self, in_dimension, out_dimension, hidden_layers, neurons_per_hidden_layer):
        super().__init__()

        if hidden_layers < 0:
            raise ValueError("hidden_layers must be >= 0")
        if hidden_layers > 0 and neurons_per_hidden_layer < 1:
            raise ValueError("neurons_per_hidden_layer must be >= 1 when there are hidden layers")

        widths = [in_dimension] + [neurons_per_hidden_layer] * hidden_layers + [out_dimension]

        self.layers = nn.ModuleList(
            nn.Linear(widths[i], widths[i + 1]) for i in range(len(widths) - 1)
        )
        self.activation = nn.ReLU()

        for layer in self.layers:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x):
        last = len(self.layers) - 1
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i != last:                 # no activation after the output layer
                x = self.activation(x)
        return x


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #

_DATASET_CACHE = {}


def load_dataset(data_file=None, data_dir=None, include_test=False):
    """Load the dataset as NumPy arrays.

    Prefers the single swept_volume_data.npz file the assignment specifies, and
    falls back to a directory of individual .npy arrays for local development.
    The testing arrays are not loaded unless include_test is set, so that code
    paths which must not see the held-out data cannot accidentally touch it.

    Results are cached, so an in-process sweep decompresses the .npz once rather
    than once per run. The returned arrays are treated as read-only.
    """
    key = (str(data_file), str(data_dir), include_test)
    if key in _DATASET_CACHE:
        return _DATASET_CACHE[key]

    wanted = ["training", "evaluation"] + (["testing"] if include_test else [])

    data_file = pathlib.Path(data_file) if data_file else DEFAULT_DATA_FILE
    if data_file.exists():
        with np.load(data_file) as handle:
            loaded = {f"{s}_{k}": handle[f"{s}_{k}"]
                      for s in wanted for k in ("features", "labels")}
        return _DATASET_CACHE.setdefault(key, (loaded, str(data_file)))

    data_dir = pathlib.Path(data_dir) if data_dir else DEFAULT_DATA_DIR
    if data_dir.is_dir():
        loaded = {f"{s}_{k}": np.load(data_dir / f"{s}_{k}.npy")
                  for s in wanted for k in ("features", "labels")}
        return _DATASET_CACHE.setdefault(key, (loaded, str(data_dir)))

    raise FileNotFoundError(
        f"could not find {data_file} or {data_dir}. Place swept_volume_data.npz in "
        f"the working directory, or pass --data / --data-dir. "
        f"During development, build the .npz with: python3 make_dataset_npz.py"
    )


def encode_features(x, encoding):
    """Optional input transformation.

    'raw'    - the 14 joint angles as given, in radians.
    'sincos' - each angle replaced by (sin, cos), giving 28 inputs. The angles
               are periodic, so this hands the network that structure directly
               instead of making it learn the wrap-around. Used by the
               self-directed investigation, not by the required sweep.
    """
    if encoding == "raw":
        return x
    if encoding == "sincos":
        return torch.cat([torch.sin(x), torch.cos(x)], dim=1)
    raise ValueError(f"unknown feature encoding: {encoding}")


def training_subset_indices(n_available, train_size):
    """Indices of the training subset for a given training-data size.

    One fixed random ordering is drawn from SUBSET_ORDER_SEED and truncated, so
    every replicate seed sees the same subset and the 1,000-example subset is
    contained in the 10,000-example subset.
    """
    if train_size > n_available:
        raise ValueError(f"train_size {train_size} exceeds {n_available} available examples")
    if train_size == n_available:
        return np.arange(n_available)
    order = np.random.default_rng(SUBSET_ORDER_SEED).permutation(n_available)
    return np.sort(order[:train_size])


def standardize_stats(tensor):
    """Per-column mean and standard deviation, guarding against zero-variance columns."""
    mean = tensor.mean(dim=0, keepdim=True)
    std = tensor.std(dim=0, unbiased=False, keepdim=True)
    std = torch.where(std > 0, std, torch.ones_like(std))
    return mean, std


# --------------------------------------------------------------------------- #
# Run bookkeeping
# --------------------------------------------------------------------------- #

def run_name(cfg):
    """Directory name uniquely identifying a configuration and its replicate seed."""
    parts = [
        f"h{cfg['hidden_layers']}",
        f"n{cfg['neurons']}",
        f"lr{cfg['lr']:g}",
        f"bs{cfg['batch_size']}",
        f"ntrain{cfg['train_size']}",
        f"ep{cfg['epochs']}",
    ]
    if cfg["feature_encoding"] != "raw":
        parts.append(cfg["feature_encoding"])
    if cfg.get("tag"):
        parts.append(cfg["tag"])
    parts.append(f"seed{cfg['seed']}")
    return "_".join(parts)


def git_commit():
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or None
    except Exception:
        return None


def environment_info(device):
    info = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "platform": platform.platform(),
        "device": str(device),
        "git_commit": git_commit(),
    }
    if device.type == "cuda":
        info["gpu_name"] = torch.cuda.get_device_name(device)
        info["cuda_build"] = torch.version.cuda
        info["cudnn"] = torch.backends.cudnn.version()
        major, minor = torch.cuda.get_device_capability(device)
        info["compute_capability"] = f"sm_{major}{minor}"
    return info


def resolve_device(requested):
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but torch.cuda.is_available() is False. "
                           "Run 'python3 verify_env.py' to diagnose.")
    return device


def sync(device):
    """Make pending GPU work complete, so wall-clock timings include it."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #

def train_one_run(cfg, verbose=True):
    """Train a single configuration with a single replicate seed.

    Returns a dict with the run summary, the per-epoch history, and the state
    dict of the best-evaluating epoch.
    """
    device = resolve_device(cfg["device"])

    raw, data_source = load_dataset(cfg.get("data_file"), cfg.get("data_dir"), include_test=False)

    # ---- assemble tensors on the target device -----------------------------
    # The whole dataset is ~6 MB, so it lives on the GPU for the entire run and
    # minibatches are gathered by indexing. This avoids per-batch host-to-device
    # copies, which would otherwise dominate the runtime of a network this small.
    x_train = torch.tensor(raw["training_features"], dtype=torch.float32, device=device)
    y_train = torch.tensor(raw["training_labels"], dtype=torch.float32, device=device)
    x_eval = torch.tensor(raw["evaluation_features"], dtype=torch.float32, device=device)
    y_eval = torch.tensor(raw["evaluation_labels"], dtype=torch.float32, device=device)

    subset = training_subset_indices(x_train.shape[0], cfg["train_size"])
    if len(subset) != x_train.shape[0]:
        index = torch.tensor(subset, dtype=torch.long, device=device)
        x_train, y_train = x_train[index], y_train[index]

    x_train = encode_features(x_train, cfg["feature_encoding"])
    x_eval = encode_features(x_eval, cfg["feature_encoding"])

    # ---- normalisation -----------------------------------------------------
    # Statistics come only from the training examples actually used in this
    # condition; evaluation data is transformed with those same statistics.
    x_mean, x_std = standardize_stats(x_train)
    y_mean, y_std = standardize_stats(y_train)

    x_train_n = (x_train - x_mean) / x_std
    y_train_n = (y_train - y_mean) / y_std
    x_eval_n = (x_eval - x_mean) / x_std
    y_eval_n = (y_eval - y_mean) / y_std

    # Converting a normalised MSE back to liters^2 is an exact rescaling, so the
    # network never has to run on unnormalised data to report real units.
    y_var = float(y_std.item()) ** 2

    # ---- seeding, then construction ----------------------------------------
    # Seeds are set before the network is built so that initialisation as well
    # as minibatch ordering is reproducible.
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    if device.type == "cuda":
        torch.cuda.manual_seed_all(cfg["seed"])

    model = NeuralNetwork(
        in_dimension=x_train_n.shape[1],
        out_dimension=y_train_n.shape[1],
        hidden_layers=cfg["hidden_layers"],
        neurons_per_hidden_layer=cfg["neurons"],
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])  # no weight decay
    loss_fn = nn.MSELoss()

    n_train = x_train_n.shape[0]
    batch_size = min(cfg["batch_size"], n_train)
    steps_per_epoch = (n_train + batch_size - 1) // batch_size

    history = {k: [] for k in ("train_loss_norm", "eval_mse_norm", "eval_rmse_liters",
                               "train_rmse_liters", "epoch_time_s")}

    best_rmse = float("inf")
    best_epoch = -1
    best_state = None
    status = "completed"
    diverged_epoch = None

    train_time = 0.0          # optimizer work only -- the figure reported as training time
    sync(device)
    wall_start = time.perf_counter()

    for epoch in range(cfg["epochs"]):
        # -------- training ---------------------------------------------------
        model.train()
        sync(device)
        epoch_start = time.perf_counter()

        order = torch.randperm(n_train, device=device)
        running = 0.0
        for start in range(0, n_train, batch_size):
            index = order[start:start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(x_train_n[index]), y_train_n[index])
            loss.backward()
            optimizer.step()
            running += float(loss.detach())

        sync(device)
        epoch_time = time.perf_counter() - epoch_start
        train_time += epoch_time
        train_loss = running / steps_per_epoch

        # -------- evaluation (excluded from train_time) ----------------------
        model.eval()
        with torch.no_grad():
            eval_mse_norm = float(loss_fn(model(x_eval_n), y_eval_n))
            # Full-training-set error, for the train-versus-evaluation diagnostic.
            # The running minibatch average above is measured while the weights
            # are still changing, so it is not comparable to the evaluation error.
            train_mse_norm = (float(loss_fn(model(x_train_n), y_train_n))
                              if cfg["track_train_metric"] else float("nan"))

        eval_rmse = (eval_mse_norm * y_var) ** 0.5
        train_rmse = (train_mse_norm * y_var) ** 0.5

        history["train_loss_norm"].append(train_loss)
        history["eval_mse_norm"].append(eval_mse_norm)
        history["eval_rmse_liters"].append(eval_rmse)
        history["train_rmse_liters"].append(train_rmse)
        history["epoch_time_s"].append(epoch_time)

        # -------- divergence check -------------------------------------------
        if not (np.isfinite(train_loss) and np.isfinite(eval_mse_norm)):
            status = "diverged"
            diverged_epoch = epoch
            if verbose:
                print(f"  epoch {epoch}: non-finite loss, stopping and recording divergence")
            break

        # -------- model selection --------------------------------------------
        if eval_rmse < best_rmse:
            best_rmse = eval_rmse
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

        if verbose and (epoch % max(1, cfg["epochs"] // 10) == 0 or epoch == cfg["epochs"] - 1):
            print(f"  epoch {epoch:>5}  train(norm) {train_loss:.5f}  "
                  f"eval RMSE {eval_rmse:8.4f} L  best {best_rmse:8.4f} L @ {best_epoch}")

    sync(device)
    total_time = time.perf_counter() - wall_start

    summary = {
        "status": status,
        "diverged_epoch": diverged_epoch,
        "epochs_completed": len(history["eval_rmse_liters"]),
        "best_eval_rmse_liters": best_rmse if np.isfinite(best_rmse) else None,
        "best_eval_mse_liters2": best_rmse ** 2 if np.isfinite(best_rmse) else None,
        "best_epoch": best_epoch,
        "train_time_s": train_time,
        "total_time_s": total_time,
        "steps_per_epoch": steps_per_epoch,
        "optimizer_steps": steps_per_epoch * len(history["eval_rmse_liters"]),
        "n_parameters": count_parameters(model),
        "n_train_used": n_train,
        "in_dimension": x_train_n.shape[1],
        "data_source": data_source,
    }

    normalization = {
        "x_mean": x_mean.cpu().numpy(), "x_std": x_std.cpu().numpy(),
        "y_mean": y_mean.cpu().numpy(), "y_std": y_std.cpu().numpy(),
    }

    return {
        "summary": summary,
        "history": history,
        "best_state": best_state,
        "normalization": normalization,
        "environment": environment_info(device),
    }


def save_run(out_dir, cfg, result):
    """Write everything needed to reproduce, analyse, and reuse a run."""
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "config.json").write_text(json.dumps(
        {"config": cfg, "environment": result["environment"]}, indent=2, sort_keys=True) + "\n")

    (out_dir / "summary.json").write_text(json.dumps(
        {"config": cfg, "environment": result["environment"], **result["summary"]},
        indent=2, sort_keys=True) + "\n")

    np.savez_compressed(
        out_dir / "history.npz",
        epoch=np.arange(len(result["history"]["eval_rmse_liters"]), dtype=np.int32),
        **{k: np.asarray(v, dtype=np.float64) for k, v in result["history"].items()},
    )

    if result["best_state"] is not None:
        # The normalisation statistics travel with the checkpoint: without them
        # the saved weights cannot be applied to raw features later.
        torch.save({
            "state_dict": result["best_state"],
            "normalization": result["normalization"],
            "config": cfg,
            "best_epoch": result["summary"]["best_epoch"],
            "best_eval_rmse_liters": result["summary"]["best_eval_rmse_liters"],
        }, out_dir / "best_model.pt")


# --------------------------------------------------------------------------- #
# Quick test
# --------------------------------------------------------------------------- #

def quick_test(args):
    """Short end-to-end check that the implementation is runnable.

    Loads the data, computes and applies training-derived normalisation, builds a
    network, trains briefly on a small subset, and runs the evaluation pipeline.
    It deliberately never loads or reports on the held-out testing set, and it
    does not replace the required experiments.
    """
    print("quick test: short training run on a small subset, evaluation pipeline only")
    print("            (the held-out testing set is never loaded)\n")

    cfg = make_config(
        experiment="quick_test", hidden_layers=2, neurons=32, lr=1e-3,
        batch_size=250, train_size=2000, epochs=20, seed=REPLICATE_SEEDS[0],
        device=args.device, feature_encoding="raw", track_train_metric=True,
        data_file=args.data, data_dir=args.data_dir, tag=None,
    )

    result = train_one_run(cfg, verbose=True)
    summary = result["summary"]

    print(f"\n  device            : {result['environment']['device']}"
          f" ({result['environment'].get('gpu_name', 'cpu')})")
    print(f"  data source       : {summary['data_source']}")
    print(f"  parameters        : {summary['n_parameters']}")
    print(f"  best eval RMSE    : {summary['best_eval_rmse_liters']:.4f} liters"
          f" at epoch {summary['best_epoch']}")
    print(f"  training time     : {summary['train_time_s']:.2f} s")

    problems = []
    if summary["status"] != "completed":
        problems.append(f"run status was {summary['status']!r}")
    if result["best_state"] is None:
        problems.append("no best checkpoint was retained")
    if not np.isfinite(summary["best_eval_rmse_liters"] or float("nan")):
        problems.append("best evaluation RMSE is not finite")
    if summary["epochs_completed"] != cfg["epochs"]:
        problems.append("did not complete the requested epochs")

    if problems:
        print("\nQUICK TEST FAILED: " + "; ".join(problems))
        return 1
    print("\nQuick test PASSED.")
    return 0


# --------------------------------------------------------------------------- #
# Command line
# --------------------------------------------------------------------------- #

def make_config(**kwargs):
    cfg = dict(kwargs)
    cfg["data_file"] = str(cfg["data_file"]) if cfg.get("data_file") else None
    cfg["data_dir"] = str(cfg["data_dir"]) if cfg.get("data_dir") else None
    return cfg


def build_parser():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    ap.add_argument("--quick-test", action="store_true",
                    help="run a short verification of the implementation and exit")

    g = ap.add_argument_group("experiment")
    g.add_argument("--experiment", default="gpu_arch",
                   help="experiment group name; becomes a subdirectory of --out (default: gpu_arch)")
    g.add_argument("--tag", default=None,
                   help="optional extra label folded into the run directory name")

    g = ap.add_argument_group("architecture")
    g.add_argument("--hidden-layers", type=int, default=2, help="number of hidden layers")
    g.add_argument("--neurons", type=int, default=128, help="neurons per hidden layer")
    g.add_argument("--feature-encoding", choices=["raw", "sincos"], default="raw",
                   help="input transformation; 'sincos' is the self-directed variant")

    g = ap.add_argument_group("training")
    g.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate")
    g.add_argument("--batch-size", type=int, default=10000, help="minibatch size")
    g.add_argument("--train-size", type=int, default=100000,
                   help="number of training examples to use (subsets are nested and fixed)")
    g.add_argument("--epochs", type=int, default=1000, help="number of training epochs")
    g.add_argument("--seed", type=int, default=0, choices=REPLICATE_SEEDS,
                   help="replicate seed")
    g.add_argument("--any-seed", action="store_true",
                   help="allow a seed outside the five required replicate seeds")
    g.add_argument("--no-train-metric", action="store_true",
                   help="skip the per-epoch full-training-set error (saves a little time)")

    g = ap.add_argument_group("system")
    g.add_argument("--device", default="auto", help="auto, cuda, cuda:N, or cpu (default: auto)")
    g.add_argument("--data", type=pathlib.Path, default=None,
                   help=f"path to the .npz dataset (default: {DEFAULT_DATA_FILE})")
    g.add_argument("--data-dir", type=pathlib.Path, default=None,
                   help=f"fallback directory of .npy arrays (default: {DEFAULT_DATA_DIR})")
    g.add_argument("--out", type=pathlib.Path, default=pathlib.Path("runs"),
                   help="root output directory (default: runs)")
    g.add_argument("--replace", action="store_true",
                   help="overwrite an existing completed run instead of refusing")
    g.add_argument("--quiet", action="store_true", help="suppress per-epoch progress output")
    return ap


def main(argv=None):
    ap = build_parser()
    # --seed is restricted to the five replicate seeds unless --any-seed is given,
    # which requires re-parsing without the choices constraint.
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--any-seed" in argv:
        for action in ap._actions:
            if action.dest == "seed":
                action.choices = None
    args = ap.parse_args(argv)

    if args.quick_test:
        return quick_test(args)

    cfg = make_config(
        experiment=args.experiment, hidden_layers=args.hidden_layers, neurons=args.neurons,
        lr=args.lr, batch_size=args.batch_size, train_size=args.train_size,
        epochs=args.epochs, seed=args.seed, device=args.device,
        feature_encoding=args.feature_encoding, track_train_metric=not args.no_train_metric,
        data_file=args.data, data_dir=args.data_dir, tag=args.tag,
    )

    out_dir = args.out / args.experiment / run_name(cfg)
    if (out_dir / "summary.json").exists() and not args.replace:
        print(f"run already complete: {out_dir}\npass --replace to overwrite it.", file=sys.stderr)
        return 2

    print(f"run: {out_dir}")
    result = train_one_run(cfg, verbose=not args.quiet)
    save_run(out_dir, cfg, result)

    s = result["summary"]
    print(f"\nstatus            : {s['status']}")
    if s["best_eval_rmse_liters"] is not None:
        print(f"best eval RMSE    : {s['best_eval_rmse_liters']:.4f} liters "
              f"(MSE {s['best_eval_mse_liters2']:.4f} L^2) at epoch {s['best_epoch']}")
    print(f"training time     : {s['train_time_s']:.2f} s "
          f"(total incl. evaluation {s['total_time_s']:.2f} s)")
    print(f"parameters        : {s['n_parameters']}   optimizer steps: {s['optimizer_steps']}")
    print(f"saved to          : {out_dir}")
    return 0 if s["status"] == "completed" else 1


if __name__ == "__main__":
    sys.exit(main())
