#!/usr/bin/env python3
"""
perf.py — Stencil benchmark runner for AD/AE evaluation.

Usage:
    # Collect GStencil/s (default 10 repeats, record max):
    python script/perf.py perf A100
    python script/perf.py perf A100 -n 3

    # Generate NCU profiling reports:
    python script/perf.py ncu A100
"""

import argparse
import ast
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
BUILD_DIR    = PROJECT_ROOT / "build"
OUTPUT_DIR   = PROJECT_ROOT / "output"
CSV_PATH     = PROJECT_ROOT / "problem_size.csv"

# ---------------------------------------------------------------------------
# Kernel name mappings
# ---------------------------------------------------------------------------

CONV_KERNEL_MAP = {
    "j2d5pt":  "star2d1r",
    "j2d9pt":  "box2d1r",
    "j2d13pt": "star2d3r",
    "j2d49pt": "box2d3r",
    "j3d7pt":  "star3d1r",
    "j3d27pt": "box3d1r",
}

# (mapped_name, gstencil_multiplier)
LORA_KERNEL_MAP = {
    "j2d5pt":  ("star2d1r", 1.0),
    "j2d9pt":  ("box2d3r",  3.0),   # compensated x3
    "j2d13pt": ("star2d3r", 1.0),
    "j2d49pt": ("box2d3r",  1.0),
    "j3d7pt":  ("star3d1r", 1.0),
    "j3d27pt": ("box3d1r",  1.0),
}

FLASH_KERNEL_MAP = {
    "j2d5pt":  "Heat-2D",
    "j2d9pt":  "Box2D9P",
    "j3d7pt":  "Heat-3D",
    "j3d27pt": "Heat-3D",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_problem_size(ps_str: str) -> list:
    return [int(x) for x in ast.literal_eval(ps_str)]


def is_3d_kernel(kernel: str) -> bool:
    return kernel.startswith("j3d") or kernel == "poisson"


def extract_gstencil(output: str) -> float | None:
    # Format 1: "GStencil/s = 123.456" or "GStencil/s: 123.456"
    m = re.search(r"GStencil/s\s*[=:]\s*([\d.]+)", output)
    if m:
        return float(m.group(1))
    # Format 2: "747.000378 GStencil/s"  (FastStencil)
    m = re.search(r"([\d.]+)\s+GStencil/s", output)
    if m:
        return float(m.group(1))
    # Format 3: EBISU tab-delimited — last token of a line with >= 10 fields
    for line in output.splitlines():
        parts = line.strip().split("\t")
        if len(parts) >= 10:
            try:
                return float(parts[-1].strip())
            except ValueError:
                pass
    return None

# ---------------------------------------------------------------------------
# Command builders
# ---------------------------------------------------------------------------

def build_faststencil_cmd(kernel: str, ps: list) -> str:
    if is_3d_kernel(kernel):
        T, NZ, NY, NX = ps
        exe = BUILD_DIR / "FastStencil" / "3dstencil" / "FastStencil_perf"
        return f"{exe} {kernel} {NZ} {NY} {NX} {T} 0"
    else:
        T, NY, NX = ps
        exe = BUILD_DIR / "FastStencil" / "2dstencil" / "FastStencil_perf"
        return f"{exe} {kernel} {NY} {NX} {T} 0"


def build_ebisu_cmd(kernel: str, ps: list) -> str:
    if is_3d_kernel(kernel):
        T, NZ, NY, NX = ps
        mapping = {
            "j3d7pt":  ("3d7pt",   "3d7pt_ebisu.exe"),
            "j3d27pt": ("3d27pt",  "3d27pt_ebisu.exe"),
            "j3d13pt": ("3d13pt",  "3d13pt_ebisu.exe"),
            "poisson": ("poisson", "poisson_ebisu.exe"),
        }
        pt_dir, exe_name = mapping.get(kernel, (kernel, f"{kernel}_ebisu.exe"))
        exe = BUILD_DIR / "EBISU" / "3dstencil" / pt_dir / exe_name
        return f"stdbuf -oL -eL {exe} {NX} {NY} {NZ} --iter={T} --warmup"
    else:
        T, NY, NX = ps
        mapping = {
            "j2d5pt":  ("2d5pt",  "2d5pt_ebisu.exe"),
            "j2d9pt":  ("2d9pt",  "2d9pt_ebisu.exe"),
            "j2ds9pt": ("2ds9pt", "2ds9pt_ebisu.exe"),
            "j2d13pt": ("2d13pt", "2d13pt_ebisu.exe"),
            "j2d25pt": ("2d25pt", "2d25pt_ebisu.exe"),
            "j2d49pt": ("2d49pt", "2d49pt_ebisu.exe"),
        }
        pt_dir, exe_name = mapping.get(kernel, (kernel, f"{kernel}_ebisu.exe"))
        exe = BUILD_DIR / "EBISU" / "2dstencil" / pt_dir / exe_name
        return f"stdbuf -oL -eL {exe} {NY} {NX} --iter={T} --warmup"


def build_convstencil_cmd(kernel: str, ps: list) -> tuple[str, float]:
    mapped = CONV_KERNEL_MAP.get(kernel)
    if mapped is None:
        raise ValueError(f"ConvStencil: unsupported kernel '{kernel}'")
    if is_3d_kernel(kernel):
        T, NZ, NY, NX = ps
        exe = BUILD_DIR / "ConvStencil" / "convstencil_3d"
        return f"{exe} {mapped} {NZ} {NY} {NX} 1", 1.0
    else:
        T, NY, NX = ps
        exe = BUILD_DIR / "ConvStencil" / "convstencil_2d"
        return f"{exe} {mapped} {NY} {NX} 1", 1.0


def build_lorastencil_cmd(kernel: str, ps: list) -> tuple[str, float]:
    entry = LORA_KERNEL_MAP.get(kernel)
    if entry is None:
        raise ValueError(f"LoRAStencil: unsupported kernel '{kernel}'")
    mapped, multiplier = entry
    if is_3d_kernel(kernel):
        T, NZ, NY, NX = ps
        exe = BUILD_DIR / "LoRAStencil" / "lorastencil_3d"
        return f"{exe} {mapped} {NZ} {NY} {NX} 1", multiplier
    else:
        T, NY, NX = ps
        exe = BUILD_DIR / "LoRAStencil" / "lorastencil_2d"
        return f"{exe} {mapped} {NY} {NX} 1", multiplier


def build_flashfftstencil_cmd(kernel: str, ps: list) -> str:
    mapped = FLASH_KERNEL_MAP.get(kernel)
    if mapped is None:
        raise ValueError(f"FlashFFTStencil: unsupported kernel '{kernel}'")
    if is_3d_kernel(kernel):
        T, NZ, NY, NX = ps
        exe = BUILD_DIR / "FlashFFTStencil" / "3d.out"
        return f"{exe} {mapped} {NZ} 1"
    else:
        T, NY, NX = ps
        exe = BUILD_DIR / "FlashFFTStencil" / "2d.out"
        return f"{exe} {mapped} {NY} 1"


def build_command(kernel: str, method: str, ps: list) -> tuple[str, float]:
    m = method.strip()
    if m == "FastStencil":
        return build_faststencil_cmd(kernel, ps), 1.0
    elif m == "EBISU":
        return build_ebisu_cmd(kernel, ps), 1.0
    elif m == "ConvStencil":
        return build_convstencil_cmd(kernel, ps)
    elif m == "LoRAStencil":
        return build_lorastencil_cmd(kernel, ps)
    elif m == "FlashFFTStencil":
        return build_flashfftstencil_cmd(kernel, ps), 1.0
    else:
        raise ValueError(f"Unknown method: '{method}'")

# ---------------------------------------------------------------------------
# Runner helpers
# ---------------------------------------------------------------------------

def run_once(cmd: str) -> tuple[str, str, bool]:
    """Run cmd once; return (stdout, stderr, timed_out)."""
    proc = subprocess.Popen(
        cmd, shell=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    try:
        stdout, stderr = proc.communicate(timeout=300)
        return stdout, stderr, False
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.communicate()
        return "", "", True

# ---------------------------------------------------------------------------
# perf mode
# ---------------------------------------------------------------------------

def run_perf(row: pd.Series, repeat: int) -> float | None:
    """Run one benchmark case `repeat` times; return the max GStencil/s or None."""
    kernel, method, ps_str, hardware = (
        row["kernel"], row["method"], row["problem size"], row["hardware"]
    )
    ps = parse_problem_size(ps_str)

    try:
        cmd, multiplier = build_command(kernel, method, ps)
    except ValueError as e:
        print(f"  [SKIP] {e}")
        return None

    print(f"\n{'='*70}")
    print(f"  kernel={kernel}  hardware={hardware}  method={method}  problem_size={ps_str}")
    print(f"  CMD: {cmd}")
    print(f"  REPEAT: {repeat}")
    print(f"{'='*70}")

    best = None
    for i in range(repeat):
        print(f"  [Run {i+1}/{repeat}]", flush=True)
        stdout, stderr, timed_out = run_once(cmd)
        combined = stdout + stderr

        if i == 0:
            print(combined)
        else:
            last_line = stdout.splitlines()[-1] if stdout.strip() else "(no output)"
            print(f"  {last_line}")

        if timed_out:
            print(f"  [ERROR] Run {i+1} timed out (>300s).")
            return None

        val = extract_gstencil(combined)
        if val is not None:
            final = val * multiplier
            if best is None or final > best:
                best = final

    if best is not None:
        if multiplier != 1.0:
            print(f"  [NOTE] x{multiplier} compensation applied.")
        print(f"  >>> GStencil/s (max of {repeat} runs) = {best:.6f}")
    else:
        print("  [ERROR] Could not parse GStencil/s from any run.")

    return best


def mode_perf(cases: pd.DataFrame, hardware: str, repeat: int) -> None:
    out_dir = OUTPUT_DIR / hardware
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "gstencil.csv"

    cases["GStencil/s"] = cases.apply(lambda row: run_perf(row, repeat), axis=1)

    out = cases[["kernel", "hardware", "method", "problem size", "GStencil/s"]]
    out.to_csv(out_csv, index=False)

    print(f"\n{'='*70}")
    print(f"[INFO] Results written to: {out_csv}")
    print(f"{'='*70}\n")

    ok  = out.dropna(subset=["GStencil/s"])
    bad = out[out["GStencil/s"].isna()]
    print(f"Summary: {len(ok)}/{len(out)} cases succeeded.\n")
    if not ok.empty:
        print(ok[["kernel", "method", "GStencil/s"]].to_string(index=False))
    if not bad.empty:
        print(f"\nFailed cases:")
        print(bad[["kernel", "method"]].to_string(index=False))

# ---------------------------------------------------------------------------
# ncu mode
# ---------------------------------------------------------------------------

def run_ncu(row: pd.Series, hardware: str) -> None:
    kernel, method, ps_str = row["kernel"], row["method"], row["problem size"]
    ps = parse_problem_size(ps_str)

    try:
        perf_cmd, _ = build_command(kernel, method, ps)
    except ValueError as e:
        print(f"  [SKIP] {e}")
        return

    rep_dir = OUTPUT_DIR / hardware / method
    rep_dir.mkdir(parents=True, exist_ok=True)
    rep_path = rep_dir / kernel   # ncu appends .ncu-rep automatically

    ncu_cmd = (
        f"ncu --set full --launch-skip 3 --launch-count 1 "
        f"-o {rep_path} {perf_cmd}"
    )

    print(f"\n{'='*70}")
    print(f"  kernel={kernel}  hardware={hardware}  method={method}  problem_size={ps_str}")
    print(f"  CMD: {ncu_cmd}")
    print(f"{'='*70}")

    stdout, stderr, timed_out = run_once(ncu_cmd)
    print(stdout + stderr)
    if timed_out:
        print("  [ERROR] ncu timed out (>300s).")
    else:
        print(f"  >>> Report: {rep_path}.ncu-rep")


def mode_ncu(cases: pd.DataFrame, hardware: str) -> None:
    cases.apply(lambda row: run_ncu(row, hardware), axis=1)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Stencil benchmark runner: collect GStencil/s or generate NCU reports."
    )
    parser.add_argument("mode", choices=["perf", "ncu"],
                        help="'perf' to collect GStencil/s, 'ncu' to profile with NCU.")
    parser.add_argument("hardware", choices=["A100", "H100"],
                        help="Target hardware platform.")
    parser.add_argument("-n", "--repeat", type=int, default=10, metavar="N",
                        help="[perf only] Repeat count per benchmark; record the max (default: 10).")
    args = parser.parse_args()
    mode, hardware, repeat = args.mode, args.hardware, args.repeat

    if not CSV_PATH.exists():
        sys.exit(f"[FATAL] Cannot find problem_size.csv at {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)
    cases = df[df["hardware"] == hardware].copy().reset_index(drop=True)

    if cases.empty:
        sys.exit(f"[FATAL] No cases found for hardware '{hardware}'")

    print(f"\n[INFO] Mode: {mode}  |  Platform: {hardware}  |  Cases: {len(cases)}")
    if mode == "perf":
        print(f"[INFO] Repeat per case: {repeat}")
    print(f"[INFO] Project root : {PROJECT_ROOT}")
    print(f"[INFO] Build dir    : {BUILD_DIR}")

    if mode == "perf":
        mode_perf(cases, hardware, repeat)
    else:
        mode_ncu(cases, hardware)


if __name__ == "__main__":
    main()
