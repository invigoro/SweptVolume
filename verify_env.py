#!/usr/bin/env python3
"""Verify that the environment satisfies the assignment's computing requirements.

Run after creating and activating the virtual environment:

    python3 verify_env.py

Exits non-zero if any required check fails. The Gymnasium environment checks
are reported but are not fatal for the neural-network assignment itself.
"""

import argparse
import platform
import sys

FAIL = []
WARN = []


def section(title):
    print(f"\n--- {title} " + "-" * max(0, 56 - len(title)))


def check(label, ok, detail="", fatal=True):
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f"  {detail}" if detail else ""))
    if not ok:
        (FAIL if fatal else WARN).append(label)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-gym", action="store_true",
                    help="skip the Gymnasium environment construction checks")
    args = ap.parse_args()

    section("Interpreter")
    print(f"  executable    : {sys.executable}")
    print(f"  version       : {platform.python_version()}")
    print(f"  machine       : {platform.machine()}")
    # pygame (a Gymnasium dependency) has no wheels beyond CPython 3.13.
    check("Python is 3.10-3.13", (3, 10) <= sys.version_info[:2] <= (3, 13),
          f"found {platform.python_version()}")
    check("running inside a virtual environment",
          sys.prefix != sys.base_prefix, sys.prefix)

    section("Required packages")
    try:
        import numpy
        print(f"  numpy         : {numpy.__version__}")
    except ImportError as exc:
        check("numpy importable", False, str(exc))
        numpy = None
    try:
        import matplotlib
        print(f"  matplotlib    : {matplotlib.__version__}")
    except ImportError as exc:
        check("matplotlib importable", False, str(exc))
    try:
        import torch
        print(f"  torch         : {torch.__version__}")
    except ImportError as exc:
        check("torch importable", False, str(exc))
        print("\nEnvironment verification FAILED (torch is required).")
        return 1

    section("CUDA / GPU")
    print(f"  built against : CUDA {torch.version.cuda}")
    print(f"  cuDNN         : {torch.backends.cudnn.version()}")
    print(f"  arch list     : {' '.join(torch.cuda.get_arch_list())}")
    check("torch.cuda.is_available()", torch.cuda.is_available())

    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        major, minor = torch.cuda.get_device_capability(0)
        sm = f"sm_{major}{minor}"
        print(f"  device 0      : {name}")
        print(f"  capability    : {sm}")
        check(f"wheel contains kernels for {sm}",
              sm in torch.cuda.get_arch_list(),
              "otherwise expect 'no kernel image is available'")
        # Actually execute a kernel: this is what catches a wheel that was
        # built without support for this GPU's compute capability.
        try:
            a = torch.rand(1024, 1024, device="cuda")
            checksum = float((a @ a).sum())
            torch.cuda.synchronize()
            check("matmul executes on the GPU", True, f"checksum {checksum:.4g}")
        except Exception as exc:
            check("matmul executes on the GPU", False, f"{type(exc).__name__}: {exc}")

    section("Gymnasium")
    try:
        import gymnasium as gym
        print(f"  gymnasium     : {gym.__version__}")
    except ImportError as exc:
        check("gymnasium importable", False, str(exc), fatal=False)
        gym = None

    if gym is not None and not args.skip_gym:
        try:
            import ale_py
            gym.register_envs(ale_py)     # current Gymnasium requires explicit ALE registration
        except ImportError as exc:
            check("ale_py importable", False, str(exc), fatal=False)
        for env_id, extra in [("CartPole-v1", "classic-control"),
                              ("LunarLander-v3", "box2d"),
                              ("ALE/Pong-v5", "atari")]:
            try:
                env = gym.make(env_id)
                env.reset()
                env.close()
                check(f"{env_id} ({extra})", True, fatal=False)
            except Exception as exc:
                check(f"{env_id} ({extra})", False,
                      f"{type(exc).__name__}: {exc}", fatal=False)

    section("Result")
    for label in WARN:
        print(f"  warning: {label}")
    if FAIL:
        for label in FAIL:
            print(f"  failed : {label}")
        print("\nEnvironment verification FAILED.")
        return 1
    print("\nEnvironment verification PASSED.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
