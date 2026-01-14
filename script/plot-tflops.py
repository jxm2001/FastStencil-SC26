#!/usr/bin/env python3
"""
Plot TFLOPS charts from output/<hardware>/gstencil.csv.
"""

import argparse
import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

HARDWARE_CONFIGS = {
    'A100': {
        'peak_tflops': 9.7,
    },
    'H100': {
        'peak_tflops': 34.0,
    }
}

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

AXIS_LABEL_FONTSIZE = 18
TICK_FONTSIZE = 16

KERNEL_ORDER = ['j2d5pt', 'j2d9pt', 'j2d13pt', 'j2d25pt', 'j2d49pt', 'j3d7pt', 'poisson', 'j3d27pt']

METHOD_ORDER = ['ConvStencil', 'LoRAStencil', 'FlashFFTStencil', 'EBISU', 'FastStencil']
METHOD_COLORS = {
    'ConvStencil': '#3498db',
    'LoRAStencil': '#2ecc71',
    'FlashFFTStencil': '#95a5a6',
    'EBISU': '#e74c3c',
    'FastStencil': '#f39c12'
}

SPEEDUP_COLOR = '#9b59b6'


def load_and_filter_data(csv_file, hardware):
    """
    Load CSV, filter by hardware, and calculate TFLOPS from GStencil/s.
    """
    df = pd.read_csv(csv_file)

    if 'GStencil/s' not in df.columns:
        print("Error: CSV must contain a 'GStencil/s' column")
        return None

    df = df.rename(columns={'GStencil/s': 'GStencil'})
    
    df_filtered = df[df['hardware'].str.upper() == hardware.upper()].copy()
    
    if len(df_filtered) == 0:
        print(f"Warning: No data found for hardware '{hardware}'")
        return None
    
    df_filtered = df_filtered[df_filtered['kernel'].isin(KERNEL_ORDER)].copy()
    
    if len(df_filtered) == 0:
        print(f"Warning: No data found for specified kernels on {hardware}")
        return None

    missing_ops = sorted(set(df_filtered['kernel']) - set(KERNEL_FP64_OPS))
    if missing_ops:
        print(f"Error: Missing FP64 operation counts for kernels: {', '.join(missing_ops)}")
        return None

    df_filtered['TFLOPS'] = (
        df_filtered['GStencil'] * df_filtered['kernel'].map(KERNEL_FP64_OPS) / 1000.0
    )
    
    return df_filtered


def plot_hardware_performance(df, hardware, output_file):
    """
    Plot performance chart for a specific hardware with dual y-axes
    Left: TFLOPS (bars), Right: Speedup (star)
    """
    import matplotlib.pyplot as plt

    if df is None or len(df) == 0:
        print(f"Skipping plot for {hardware} - no data available")
        return
    
    hw_config = HARDWARE_CONFIGS[hardware]
    peak_tflops = hw_config['peak_tflops']
    
    fig, ax1 = plt.subplots(figsize=(16, 4))
    ax2 = ax1.twinx()
    
    kernels_present = [k for k in KERNEL_ORDER if k in df['kernel'].values]
    kernel_methods = {}
    
    for kernel in kernels_present:
        kernel_data = df[df['kernel'] == kernel]
        methods_with_data = []
        for method in METHOD_ORDER:
            method_data = kernel_data[kernel_data['method'] == method]
            if len(method_data) > 0:
                gstencil = method_data['GStencil'].iloc[0]
                tflops = method_data['TFLOPS'].iloc[0]
                if pd.notna(gstencil) or pd.notna(tflops):
                    methods_with_data.append(method)
        kernel_methods[kernel] = methods_with_data
    
    bar_width = 1.0
    group_gap = 1.0
    
    kernel_group_positions = []
    current_x = 0
    
    for kernel in kernels_present:
        n_methods = len(kernel_methods[kernel])
        group_center = current_x + (n_methods * bar_width) / 2
        kernel_group_positions.append(group_center)
        current_x += n_methods * bar_width + group_gap
    
    method_bar_positions = {method: [] for method in METHOD_ORDER}
    method_tflops_values = {method: [] for method in METHOD_ORDER}
    method_speedup_values = {method: [] for method in METHOD_ORDER}
    
    current_x = 0
    for kernel in kernels_present:
        methods_for_kernel = kernel_methods[kernel]
        
        kernel_data = df[df['kernel'] == kernel]
        faststencil_data = kernel_data[kernel_data['method'] == 'FastStencil']
        if len(faststencil_data) > 0 and pd.notna(faststencil_data['GStencil'].iloc[0]):
            faststencil_gstencil = faststencil_data['GStencil'].iloc[0]
        else:
            faststencil_gstencil = None
        
        for local_idx, method in enumerate(methods_for_kernel):
            method_data = df[(df['kernel'] == kernel) & (df['method'] == method)]
            gstencil = method_data['GStencil'].iloc[0]
            tflops_val = method_data['TFLOPS'].iloc[0]
            
            if pd.isna(tflops_val):
                tflops_val = 0
            
            if method != 'FastStencil' and pd.notna(gstencil) and gstencil > 0 and faststencil_gstencil is not None:
                speedup_val = faststencil_gstencil / gstencil
            else:
                speedup_val = None
            
            bar_pos = current_x + (local_idx + 0.5) * bar_width
            
            method_bar_positions[method].append(bar_pos)
            method_tflops_values[method].append(tflops_val)
            method_speedup_values[method].append(speedup_val)
        
        current_x += len(methods_for_kernel) * bar_width + group_gap
    
    for method in METHOD_ORDER:
        if len(method_bar_positions[method]) > 0:
            positions = np.array(method_bar_positions[method])
            tflops_vals = np.array(method_tflops_values[method])
            
            ax1.bar(positions, tflops_vals,
                   bar_width,
                   label=method,
                   color=METHOD_COLORS[method],
                   alpha=0.8,
                   edgecolor='black',
                   linewidth=0.5)
    
    all_speedup_positions = []
    all_speedup_values = []
    
    for method in METHOD_ORDER:
        if method == 'FastStencil':
            continue
        
        if len(method_bar_positions[method]) > 0:
            positions = np.array(method_bar_positions[method])
            speedup_vals = method_speedup_values[method]
            
            for pos, speedup in zip(positions, speedup_vals):
                if speedup is not None:
                    all_speedup_positions.append(pos)
                    all_speedup_values.append(speedup)
    
    # Values above 8x are compressed into the visual [8, 16] interval and annotated.
    if len(all_speedup_positions) > 0:
        positions_normal = []
        speedups_normal = []
        positions_high = []
        speedups_high = []
        speedups_high_actual = []
        
        for pos, speedup in zip(all_speedup_positions, all_speedup_values):
            if speedup <= 8:
                positions_normal.append(pos)
                speedups_normal.append(speedup)
            else:
                positions_high.append(pos)
                speedups_high_actual.append(speedup)
                compressed = 8 * ((speedup / 8) ** 0.23)
                speedups_high.append(compressed)
        
        if len(positions_normal) > 0:
            ax2.scatter(positions_normal, speedups_normal,
                       color=SPEEDUP_COLOR,
                       s=100,
                       marker='*',
                       edgecolors='black',
                       linewidths=0.5,
                       zorder=5,
                       label='Speedup')
        
        if len(positions_high) > 0:
            ax2.scatter(positions_high, speedups_high,
                       color=SPEEDUP_COLOR,
                       s=100,
                       marker='*',
                       edgecolors='black',
                       linewidths=0.5,
                       zorder=5,
                       label='Speedup' if len(positions_normal) == 0 else '')
            
            for pos, compressed, actual in zip(positions_high, speedups_high, speedups_high_actual):
                ax2.annotate(f'{actual:.1f}×',
                           xy=(pos, compressed),
                           xytext=(0, 8),
                           textcoords='offset points',
                           ha='center',
                           va='bottom',
                           fontsize=TICK_FONTSIZE,
                           color=SPEEDUP_COLOR,
                           fontweight='bold')
    
    tflops_max = peak_tflops * 1.1
    tflops_ticks = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    tflops_values = [tick * peak_tflops for tick in tflops_ticks]
    tflops_labels = []
    
    for tick, value in zip(tflops_ticks, tflops_values):
        if tick == 0:
            tflops_labels.append('0')
        elif tick == 1.0:
            tflops_labels.append(f'{value:.1f}\n(100%)')
        else:
            percent = int(tick * 100)
            tflops_labels.append(f'{value:.1f}\n({percent}%)')
    
    ax1.set_yticks(tflops_values)
    ax1.set_yticklabels(tflops_labels, fontsize=TICK_FONTSIZE)
    ax1.set_ylim(0, tflops_max)
    
    ax1.axhline(y=peak_tflops, 
                color='red', 
                linestyle='--', 
                linewidth=2,
                zorder=4)
    
    ax1.text(-group_gap*0.9, peak_tflops * 1.01, 
             'CUDA Core FP64 Peak', 
             verticalalignment='bottom',
             horizontalalignment='left',
             fontsize=TICK_FONTSIZE,
             color='red',
             fontweight='bold')

    ax1.text(current_x - group_gap + group_gap * 0.9, peak_tflops * 1.01, 
             f'NVIDIA {hardware}', 
             verticalalignment='bottom',
             horizontalalignment='right',
             fontsize=TICK_FONTSIZE,
             color='black',
             fontweight='bold')
    
    ax2.set_yscale('log', base=2)
    
    speedup_ticks = [1, 2, 4, 8, 16]
    speedup_labels = ['1×', '2×', '4×', '8×', r'$\infty$']
    
    ax2.set_yticks(speedup_ticks)
    ax2.set_yticklabels(speedup_labels, fontsize=TICK_FONTSIZE)
    labels = ax2.get_yticklabels()
    labels[-1].set_fontsize(TICK_FONTSIZE * 1.5)
    
    # Align 1x with the 20% TFLOPS tick and infinity with the FP64 peak line.
    log_range = 4 / (1.0 / 1.1 - 0.2 / 1.1)
    log_bottom = 0 - (0.2 / 1.1) * log_range
    log_top = log_bottom + log_range
    
    ax2.set_ylim(2**log_bottom, 2**log_top)
    
    ax1.set_ylabel('TFLOPS', fontsize=AXIS_LABEL_FONTSIZE, fontweight='bold', color='black')
    ax1.tick_params(axis='y', labelcolor='black')
    
    ax2.set_ylabel('Speedup', fontsize=AXIS_LABEL_FONTSIZE, fontweight='bold', color='black')
    ax2.tick_params(axis='y', labelcolor='black')
    
    ax1.set_xticks(kernel_group_positions)
    ax1.set_xticklabels(kernels_present, rotation=0, ha='center', fontsize=TICK_FONTSIZE)
    
    ax1.set_xlim(-group_gap, current_x - group_gap + group_gap)
    
    ax1.set_axisbelow(True)
    ax1.yaxis.grid(True, alpha=0.7, linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    
    output_path = Path(output_file).with_suffix('.png')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to: {output_path}")
    
    plt.close(fig)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot TFLOPS charts from output/<hardware>/gstencil.csv."
    )
    parser.add_argument(
        "hardware",
        choices=sorted(HARDWARE_CONFIGS),
        help="Target hardware to plot."
    )
    return parser.parse_args()

def main():
    args = parse_args()
    hardware = args.hardware

    csv_file = Path(ROOT_DIR) / 'output' / hardware / 'gstencil.csv'
    output_dir = Path(ROOT_DIR) / 'output' / hardware / 'images'
    
    if not csv_file.exists():
        print(f"Error: File not found: {csv_file}")
        sys.exit(1)
    
    print("="*80)
    print(f"Loading data from: {csv_file}")
    print("="*80)
    
    df = load_and_filter_data(csv_file, hardware)
    if df is None:
        sys.exit(1)

    output_file = output_dir / f"performance_{hardware}"
    plot_hardware_performance(df, hardware, output_file)
    
if __name__ == "__main__":
    main()
