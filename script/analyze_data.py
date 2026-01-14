#!/usr/bin/env python3
"""
Merge output/<hardware>/gstencil.csv and output/<hardware>/ncu.csv,
then write output/<hardware>/analysis.csv.
"""

import argparse
import ast
import math
import os
import sys
from pathlib import Path

import pandas as pd

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

KERNEL_FP64_OPS = {
    'j2d5pt': 10,
    'j2d9pt': 18,
    'j2d13pt': 26,
    'j2d25pt': 50,
    'j2d49pt': 98,
    'j3d7pt': 14,
    'poisson': 38,
    'j3d27pt': 54,
}

FP64_INSTRUCTION_COLS = [
    'Executed DADD Instructions [inst]',
    'Executed DMUL Instructions [inst]',
    'Executed DFMA Instructions [inst]',
]
TENSOR_CORE_INSTRUCTION_COL = 'Executed DMMA Instructions [inst]'


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge stencil performance and NCU metrics into an analysis CSV."
    )
    parser.add_argument(
        "hardware",
        choices=["A100", "H100"],
        help="Target hardware platform."
    )
    return parser.parse_args()


def load_and_merge_data(hardware):
    output_dir = Path(ROOT_DIR) / 'output' / hardware
    gstencil_csv = output_dir / 'gstencil.csv'
    ncu_csv = output_dir / 'ncu.csv'

    if not gstencil_csv.exists():
        print(f"Error: File not found: {gstencil_csv}", file=sys.stderr)
        sys.exit(1)

    if not ncu_csv.exists():
        print(f"Error: File not found: {ncu_csv}", file=sys.stderr)
        sys.exit(1)

    gstencil_df = pd.read_csv(gstencil_csv)
    ncu_df = pd.read_csv(ncu_csv)

    gstencil_df = gstencil_df.rename(columns={'GStencil/s': 'GStencil'})
    df = pd.merge(
        gstencil_df,
        ncu_df,
        on=['method', 'hardware', 'kernel'],
        how='inner'
    )

    if len(df) == 0:
        print(f"Error: No merged data found for hardware '{hardware}'", file=sys.stderr)
        sys.exit(1)

    df['GFLOPS'] = df['GStencil'] * df['kernel'].map(KERNEL_FP64_OPS)
    df['Executed FP64 OP per stencil'] = df.apply(
        calculate_fp64_op_per_stencil,
        axis=1
    )
    df['Computational Infation Factor'] = df.apply(
        calculate_computational_infation_factor,
        axis=1
    )
    df['Measured AI'] = df.apply(calculate_measured_ai, axis=1)
    df['Effective AI'] = df['Measured AI'] / df['Computational Infation Factor']

    return df


def calculate_fp64_op_per_stencil(row):
    problem_size = ast.literal_eval(row['problem size'])
    total_stencil = math.prod(problem_size)
    cuda_core_instructions = 0

    for col in FP64_INSTRUCTION_COLS:
        if col in row and pd.notna(row[col]):
            cuda_core_instructions += row[col]

    tensor_core_instructions = 0
    if TENSOR_CORE_INSTRUCTION_COL in row and pd.notna(row[TENSOR_CORE_INSTRUCTION_COL]):
        tensor_core_instructions = row[TENSOR_CORE_INSTRUCTION_COL]

    executed_tensor_core_instructions = tensor_core_instructions / total_stencil
    executed_cuda_core_instructions = cuda_core_instructions / total_stencil

    # One DMMA884 instruction maps to 512 FP64 OPs; one DFMA maps to 64 FP64 OPs.
    return executed_tensor_core_instructions * 512 + executed_cuda_core_instructions * 64


def calculate_computational_infation_factor(row):
    if row['kernel'] not in KERNEL_FP64_OPS:
        return pd.NA

    return (
        row['Executed FP64 OP per stencil']
        / KERNEL_FP64_OPS[row['kernel']]
    )


def calculate_measured_ai(row):
    measured_ai = row['Shared Pipe Utilization [%]'] / row['L1/TEX Cache Throughput [%]']
    if row['hardware'] == 'A100':
        measured_ai *= 0.5

    return measured_ai


def build_analysis_table(df):
    df = df.copy()
    df['TFLOPS'] = df['GFLOPS'] / 1000.0

    columns = [
        'kernel',
        'hardware',
        'method',
        'Shared Pipe Utilization [%]',
        'L1/TEX Cache Throughput [%]',
        'L2 Cache Throughput [%]',
        'DRAM Throughput [%]',
        'Computational Infation Factor',
        'Measured AI',
        'Effective AI',
        'GStencil',
        'TFLOPS',
    ]
    output_df = df[columns].copy()
    output_df = output_df.rename(columns={
        'Shared Pipe Utilization [%]': 'Shared Pipe. (%)',
        'L1/TEX Cache Throughput [%]': 'L1 Cache (%)',
        'L2 Cache Throughput [%]': 'L2 Cache (%)',
        'DRAM Throughput [%]': 'DRAM (%)',
        'GStencil': 'GStencil/s',
    })

    return output_df


def main():
    args = parse_args()
    hardware = args.hardware

    df = load_and_merge_data(hardware)
    analysis_df = build_analysis_table(df)

    output_csv = Path(ROOT_DIR) / 'output' / hardware / 'analysis.csv'
    analysis_df.to_csv(output_csv, index=False)


if __name__ == "__main__":
    main()
