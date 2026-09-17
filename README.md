# Swept Volume — Assignment 1: Neural Networks

Predicting the swept volume (in liters) of a seven-jointed robot moving between a
starting and an ending joint configuration, using a fully connected neural network
in PyTorch.

- **Course number:** 591
- **Assignment choice:** GPU — Network Architecture Analysis
- **Name:** Timothy Wells

> Status: environment setup complete. Experiment code (`neural_network.py`,
> `sweep.py`, `aggregate.py`, `plots.py`, `final_test.py`) is not written yet; the
> sections describing how to run experiments will be filled in as those land.

---

## 1. Environment setup

The environment below was built and verified on Ubuntu 26.04 with an RTX 5090.
The same environment definition works unchanged on the RTX 5080 (laptop) and the
RTX 3070 — see [Why these versions](#4-why-these-versions).

### Verified configuration

| Component | Version |
|---|---|
| OS | Ubuntu 26.04.1 LTS |
| Python | 3.13.12 |
| PyTorch | 2.13.0+cu129 (CUDA 12.9 build) |
| NumPy | 2.5.3 |
| Matplotlib | 3.11.2 |
| Gymnasium | 1.3.0 (atari, box2d, classic-control, other) |
| NVIDIA driver | 580.178.04 (supports CUDA 13.0) |
| GPU | NVIDIA GeForce RTX 5090, compute capability `sm_120` |

### 1.1 Prerequisites

An NVIDIA GPU with a driver new enough for CUDA 12.9 (**driver >= 525**, and in
practice >= 570 for Blackwell cards). Check with:

```bash
nvidia-smi
```

The CUDA toolkit does **not** need to be installed system-wide — the PyTorch
wheels bundle their own CUDA runtime libraries. `nvcc` is not required.

### 1.2 Create the virtual environment (Python 3.13)

**Python 3.13 is required** — not 3.14. See
[Why these versions](#4-why-these-versions) for the reason.

Pick whichever option matches the machine:

<details open>
<summary><b>Option A — distributions where <code>python3.13-venv</code> is available</b> (most systems)</summary>

```bash
cd /path/to/SweptVolume
python3.13 -m venv venv_gpu
source venv_gpu/bin/activate
```

If this fails with `No module named 'ensurepip'`, install the venv package
(`sudo apt install python3.13-venv` on Debian/Ubuntu) or use Option B.
</details>

<details open>
<summary><b>Option B — Ubuntu 26.04 and other systems without a <code>python3.13-venv</code> package</b></summary>

Ubuntu 26.04 ships Python 3.14 as its system Python; `python3.13` exists but its
`ensurepip` module is not packaged and there is no `python3.13-venv` package, so
`python3.13 -m venv` cannot bootstrap pip. Use [uv](https://docs.astral.sh/uv/),
which needs no root access:

```bash
# install uv once, if not already present
curl -LsSf https://astral.sh/uv/install.sh | sh

cd /path/to/SweptVolume
uv venv --python 3.13 --seed venv_gpu     # --seed installs pip into the venv
source venv_gpu/bin/activate
```

`uv venv --python 3.13` reuses a system Python 3.13 if one exists and otherwise
downloads a standalone build. To force the system interpreter, pass the path
explicitly: `uv venv --python /usr/bin/python3.13 --seed venv_gpu`.
</details>

Confirm the interpreter before installing anything:

```bash
which python3
python3 -c "import platform, sys; print(sys.executable); print(platform.machine())"
```

The path must be inside `venv_gpu`. If the shell prompt shows both `(venv_gpu)`
and `(base)`, run `conda deactivate` until the Conda marker is gone — do not run
a venv nested inside an active Conda environment.

### 1.3 Install the packages

```bash
python3 -m pip install --upgrade pip setuptools wheel
python3 -m pip install -r requirements.txt
```

`requirements.txt` carries its own `--extra-index-url` line pointing at the
PyTorch CUDA 12.9 wheel index, so no extra command-line flags are needed. The
download is roughly 3 GB (the `torch` wheel alone is 930 MB).

To reproduce the exact transitive dependency set instead, use the full lock:

```bash
python3 -m pip install --extra-index-url https://download.pytorch.org/whl/cu129 \
    -r requirements-lock.txt
```

> **Do not** copy a CUDA-version-specific install command from an older handout
> or another computer; CUDA/PyTorch pairings change over time. If `cu129` is
> retired, choose the current CUDA build from
> <https://pytorch.org/get-started/locally/> and update the index URL and the
> `+cuXXX` version tag in `requirements.txt` together.

### 1.4 Verify

```bash
python3 verify_env.py
```

This checks the interpreter version, that you are inside a virtual environment,
the required packages, CUDA availability, that the installed wheel contains
kernels for this GPU's compute capability, that a matmul actually executes on the
GPU, and that the Gymnasium classic-control / Box2D / Atari environments
construct. It exits non-zero on failure. Add `--skip-gym` to skip the Gymnasium
checks. Expected output ends with:

```
Environment verification PASSED.
```

### 1.5 Dataset

The graded submission expects `swept_volume_data.npz` in the extracted submission
directory, and all code uses relative paths. For local development the six
`.npy` arrays live in `source_data/swept_volume_data/`; pack them into the
expected file with:

```bash
python3 make_dataset_npz.py          # writes ./swept_volume_data.npz
```

Neither `source_data/` nor `swept_volume_data.npz` is committed or included in
the submission ZIP — the grader supplies the dataset.

The arrays are:

| Array | Shape | dtype |
|---|---|---|
| `training_features` | (100000, 14) | float32 |
| `training_labels` | (100000, 1) | float32 |
| `evaluation_features` | (10000, 14) | float32 |
| `evaluation_labels` | (10000, 1) | float32 |
| `testing_features` | (10000, 14) | float32 |
| `testing_labels` | (10000, 1) | float32 |

Features `[:, :7]` are the starting joint angles in radians and `[:, 7:]` the
ending joint angles. Labels are swept volume in liters.

---

## 2. Running the experiments

> To be completed when the experiment code lands. This section will document:
> the exact command to run one configuration with one seed, the exact quick-test
> command, how to aggregate independent runs into the full sweep, how to run
> final held-out testing after configuration selection, and a description of the
> output files.

---

## 3. Troubleshooting

**`torch.cuda.is_available()` is False.**
Usually a CPU-only wheel. Check `python3 -c "import torch; print(torch.__version__)"` —
a CUDA build's version ends in `+cuXXX`. A plain `2.13.0`, or anything tagged
`+cpu` or `+debian`, is CPU-only; reinstall from the CUDA index. Also confirm
`nvidia-smi` works and that you are not inside a container without GPU passthrough.

**`CUDA error: no kernel image is available for execution on the device`.**
The wheel has no kernels for this GPU. Compare `torch.cuda.get_device_capability(0)`
against `torch.cuda.get_arch_list()`. Blackwell cards (RTX 50-series, `sm_120`)
need a CUDA 12.8 or newer build. `verify_env.py` reports both lists.

**`No module named 'ensurepip'` when creating the venv.**
See Option B in [1.2](#12-create-the-virtual-environment-python-313).

**`RuntimeError: Unable to run "sdl-config"` while installing Gymnasium.**
pip is trying to build `pygame` from source because no wheel matches your Python
version — this is exactly what happens on Python 3.14. Use Python 3.13.

**`pip install torch` reports "No matching distribution found".**
The Python version has no matching wheel on the chosen index, or the
`--extra-index-url` was dropped. Check `python3 --version` and that the index URL
in `requirements.txt` still exists.

**The prompt shows both `(venv_gpu)` and `(base)`.**
Two environment systems are active. Run `conda deactivate` until the Conda marker
is gone. If the venv was created while an incompatible Conda Python was active,
delete and recreate it.

**Running on a remote machine.**
Use a persistent terminal so runs survive a dropped connection:

```bash
ssh <username>@<hostname>
screen            # or: tmux
# start the experiment, then detach with Ctrl-a d
screen -r         # reattach later
```

---

## 4. Why these versions

**Python 3.13, not 3.14.** Ubuntu 26.04's system Python is 3.14, and PyTorch does
publish `cp314` CUDA wheels — but `pygame`, pulled in by Gymnasium's
`classic-control` and `box2d` extras, publishes wheels only up to `cp313`. On
3.14 pip falls back to compiling pygame from source and fails with
`Unable to run "sdl-config"` unless SDL development headers are installed
system-wide. Python 3.13 installs everything from binary wheels with no system
dependencies. (The install ultimately resolves `pygame-ce`, the community fork,
which does ship `cp313` wheels.)

**CUDA 12.9 build (`cu129`).** It has to cover all three machines this project
runs on:

| GPU | Architecture | Compute capability |
|---|---|---|
| RTX 3070 | Ampere | `sm_86` |
| RTX 5080 (laptop) | Blackwell | `sm_120` |
| RTX 5090 | Blackwell | `sm_120` |

Blackwell requires CUDA 12.8 or newer, which rules out older wheels. The
installed wheel's arch list is
`sm_75 sm_80 sm_86 sm_90 sm_100 sm_120`, which covers both `sm_86` and `sm_120`,
so **one environment definition works on all three machines**. A CUDA 13 build
(`cu130`) would also work here but requires a driver >= 580 on every machine;
`cu129` keeps the driver floor lower.

**`torch.cuda.get_arch_list()` matters more than the CUDA version string.** A
wheel built for CUDA 12.9 that lacks `sm_120` kernels still fails on a 5090 at
the first kernel launch, not at import. That is why `verify_env.py` runs an
actual matmul rather than only checking `torch.cuda.is_available()`.

---

## 5. Measured performance (RTX 5090)

Benchmarked with the full 100,000-example training set resident in GPU memory,
batch size 10,000 (10 optimizer steps per epoch), Adam, plus a full evaluation
pass every epoch:

| Architecture | Time / 1000 epochs |
|---|---|
| 1 hidden layer, 32 neurons | ~2.6 s |
| 3 hidden layers, 256 neurons | ~3.9 s |

The required sweep is 12 architecture conditions x 5 replicate seeds = 60 runs,
so the full sweep is on the order of **3–4 minutes** of GPU time. These networks
are small enough that kernel-launch overhead dominates, which is why depth and
width barely change the wall time. Keeping the whole dataset on the GPU and
indexing it with `torch.randperm` — rather than using a `DataLoader` — is what
makes this possible.

Per the assignment, all timing results reported in the report must come from a
single machine; the RTX 5090 is the designated timing machine and the hardware
must be documented in the report.

---

## 6. Repository layout

```
.
|-- README.md                  this file
|-- requirements.txt           top-level pinned dependencies
|-- requirements-lock.txt      full transitive lock of the verified environment
|-- verify_env.py              environment verification script
|-- source_data/               local copy of the dataset (.npy), not committed
|-- venv_gpu/                  virtual environment, not committed
`-- swept_volume_data.npz      built by make_dataset_npz.py, not committed
```

---

## 7. Acknowledgment

AI assistance used in this project is documented in `ACKNOWLEDGMENT.md`, as
required by the assignment.
