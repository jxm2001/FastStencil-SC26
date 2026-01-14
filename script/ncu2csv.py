#!/usr/bin/env python3
"""
Parse Nsight Compute reports from output/<hardware>/<method>/*.ncu-rep
and write output/<hardware>/ncu.csv.
"""

import argparse
import os
import sys
from pathlib import Path
import pandas as pd

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

try:
    import ncu_report
except ImportError:
    print("Error: Unable to import ncu_report module")
    print("Please ensure NVIDIA Nsight Compute extras/python directory is added to PYTHONPATH")
    print("Example: export PYTHONPATH=/usr/local/cuda/nsight-compute-2025.2.1/extras/python:$PYTHONPATH")
    sys.exit(1)


METRIC_NAME_MAPPING = {
    "sm__throughput.avg.pct_of_peak_sustained_elapsed": "Compute (SM) Throughput [%]",
    "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed": "Memory Throughput [%]",
    "l1tex__throughput.avg.pct_of_peak_sustained_active": "L1/TEX Cache Throughput [%]",
    "lts__throughput.avg.pct_of_peak_sustained_elapsed": "L2 Cache Throughput [%]",
    "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed": "DRAM Throughput [%]",
    "sm__pipe_shared_cycles_active.avg.pct_of_peak_sustained_elapsed": "Shared Pipe Utilization [%]",
    "sm__pipe_fp64_cycles_active.avg.pct_of_peak_sustained_elapsed": "FP64 Pipe Utilization [%]",
    "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed": "Tensor Pipe Utilization [%]",
    "smsp__inst_executed.sum": "Executed Instructions [inst]",
}


def extract_kernel_metrics(action, metrics_to_extract=None):
    kernel_data = {}
    
    if metrics_to_extract is None:
        metrics_to_extract = action.metric_names()
    
    for metric_name in metrics_to_extract:
        metric = action.metric_by_name(metric_name)
        if metric is not None:
            display_name = METRIC_NAME_MAPPING.get(metric_name, metric_name)
            value = metric.value()
            
            if metric.num_instances() > 1:
                if metric.has_correlation_ids():
                    corr_ids_metric = metric.correlation_ids()
                    
                    instance_dict = {}
                    for i in range(metric.num_instances()):
                        key = corr_ids_metric.value(i)
                        val = metric.value(i)
                        instance_dict[key] = val
                    
                    kernel_data[display_name] = {
                        "type": "instanced_with_correlation",
                        "instances": instance_dict,
                        "num_instances": metric.num_instances()
                    }
                else:
                    instance_values = []
                    for i in range(metric.num_instances()):
                        if metric.has_value(i):
                            instance_values.append(metric.value(i))
                    
                    kernel_data[display_name] = {
                        "type": "instanced_without_correlation",
                        "values": instance_values,
                        "num_instances": metric.num_instances()
                    }
            else:
                kernel_data[display_name] = {
                    "type": "single_value",
                    "value": value,
                }
    
    return kernel_data


def process_ncu_report(report_path, base_dir, hardware):
    """
    Process a single NCU report file
    Extract method and kernel from directory structure
    """
    print(f"Processing report: {report_path}")
    
    rel_path = report_path.relative_to(base_dir)
    parts = rel_path.parts
    
    method = parts[0]
    
    filename = report_path.stem
    if '-' in filename:
        kernel = filename.rsplit('-', 1)[1]
    else:
        kernel = filename
    
    try:
        context = ncu_report.load_report(str(report_path))
    except Exception as e:
        print(f"Failed to load report: {e}")
        return None
    
    if context.num_ranges() == 0:
        print(f"No ranges found in report: {report_path}")
        return None
    
    current_range = context.range_by_idx(0)
    
    if current_range.num_actions() == 0:
        print(f"No actions found in first range: {report_path}")
        return None
    
    action = current_range.action_by_idx(0)
    
    common_metrics = [
        "sm__throughput.avg.pct_of_peak_sustained_elapsed",
        "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
        "l1tex__throughput.avg.pct_of_peak_sustained_active",
        "lts__throughput.avg.pct_of_peak_sustained_elapsed",
        "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
        "sm__pipe_shared_cycles_active.avg.pct_of_peak_sustained_elapsed",
        "sm__pipe_fp64_cycles_active.avg.pct_of_peak_sustained_elapsed",
        "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed",
        "smsp__inst_executed.sum",
        "sass__inst_executed_per_opcode",
    ]
    
    kernel_data = extract_kernel_metrics(action, common_metrics)
    
    sass_metric_key = None
    for key in kernel_data.keys():
        if "sass__inst_executed_per_opcode" in str(key):
            sass_metric_key = key
            break
    
    if sass_metric_key and kernel_data[sass_metric_key].get("type") == "instanced_with_correlation":
        instances = kernel_data[sass_metric_key].get("instances", {})
        for opcode, count in instances.items():
            instruction_name = f"Executed {opcode} Instructions [inst]"
            kernel_data[instruction_name] = {
                "type": "single_value",
                "value": count,
            }
        del kernel_data[sass_metric_key]
    
    row = {
        "method": method,
        "hardware": hardware,
        "kernel": kernel
    }
    
    for key, value in kernel_data.items():
        if isinstance(value, dict):
            metric_type = value.get("type", "unknown")
            
            if metric_type == "single_value":
                row[key] = value.get("value")
            
            elif metric_type == "instanced_with_correlation":
                instances = value.get("instances", {})
                for inst_key, inst_val in instances.items():
                    row[f"{key}_{inst_key}"] = inst_val
            
            elif metric_type == "instanced_without_correlation":
                values = value.get("values", [])
                if values:
                    row[key] = sum(values) / len(values)
    
    return row


def parse_args():
    parser = argparse.ArgumentParser(
        description="Parse Nsight Compute reports under output/<hardware>."
    )
    parser.add_argument(
        "hardware",
        choices=["A100", "H100"],
        help="Target hardware platform."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    hardware = args.hardware
    
    base_dir = Path(ROOT_DIR) / "output" / hardware
    
    if not base_dir.is_dir():
        print(f"Error: {base_dir} is not a directory")
        sys.exit(1)
    
    report_files = list(base_dir.rglob("*.ncu-rep"))
    print(f"Found {len(report_files)} report files in directory {base_dir}")
    
    if not report_files:
        print("No .ncu-rep files found")
        return
    
    all_rows = []
    for report_file in report_files:
        row = process_ncu_report(report_file, base_dir, hardware)
        if row:
            all_rows.append(row)
    
    if not all_rows:
        print("No reports were successfully processed")
        return
    
    df = pd.DataFrame(all_rows)
    
    first_cols = ["method", "hardware", "kernel"]
    other_cols = [col for col in df.columns if col not in first_cols]
    df = df[first_cols + other_cols]
    
    output_csv = base_dir / "ncu.csv"
    
    df.to_csv(output_csv, index=False)
    print(f"Results saved to: {output_csv}")
    
    print("\n" + "="*80)
    print("Report Summary")
    print("="*80)
    print(f"Total kernels processed: {len(df)}")
    print(f"\nMethods: {df['method'].unique().tolist()}")
    print(f"Hardware: {df['hardware'].unique().tolist()}")
    print(f"Kernels: {df['kernel'].unique().tolist()}")
    print(f"\nDataFrame shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")


if __name__ == "__main__":
    main()
