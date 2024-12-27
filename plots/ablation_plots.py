"""
Module for creating ablation study plots for clustering algorithms.

This module provides functionality to analyze and visualize the performance of
different clustering methods across various parameters and privacy settings.
It focuses on plotting Normalized Intra-cluster Variance (NICV) against different
parameters while considering different privacy mechanisms and dataset characteristics.
"""

import os
from collections import defaultdict
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from configs import exp_parameter_dict

# Constants
QUALITY_KEY = 'Normalized Intra-cluster Variance (NICV)'
FIGURE_SIZE = (15, 15)
COLOR_MAP_NAME = 'tab10'
MARKER_STYLES = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
LINE_STYLES = ['-', '--', '-.', ':']


def extract_meta(dataset_name: str) -> Tuple[int, int]:
    if "synth" in dataset_name.lower():
        _, k, d, _ = dataset_name.split("_")
        return int(k), int(d)
    raise ValueError(f"Unknown dataset format: {dataset_name}")


def create_style_mappings(k_values: set, d_values: set, dps: np.ndarray) -> Tuple[Dict, Dict, Dict]:
    color_map = plt.colormaps[COLOR_MAP_NAME]
    color_mapping = {d: color_map(i) for i, d in enumerate(sorted(d_values))}
    marker_mapping = {k: MARKER_STYLES[i % len(MARKER_STYLES)]
                      for i, k in enumerate(sorted(k_values))}
    line_mapping = {dp: LINE_STYLES[i % len(LINE_STYLES)]
                    for i, dp in enumerate(dps)}

    return color_mapping, marker_mapping, line_mapping


def normalize_data(data: pd.DataFrame, dps: np.ndarray) -> Tuple[pd.DataFrame, float]:
    if "laplace" in dps:
        norm_mask = (data['dp'] == "laplace") & (data['method'] == "none")
    else:
        norm_mask = (data['dp'] == "none") & (data['method'] == "none")

    norm_value = data[norm_mask][QUALITY_KEY].values[0]
    data = data[data['method'] != "none"]
    return data, norm_value


def calculate_averages(xs: Dict[str, List], ys: Dict[str, List], all_xs: np.ndarray,
                       pre: str, suff: str) -> List[float]:
    avg_ys = []
    for x in all_xs:
        y_values = [ys[_dataset][i]
                    for _dataset in xs
                    if x in xs[_dataset]
                    for i, x_val in enumerate(xs[_dataset])
                    if x_val == x and _dataset.startswith(pre) and _dataset.endswith(suff)]
        avg_ys.append(np.mean(y_values) if y_values else np.nan)
    return avg_ys


def create_legend_handles(color_mapping: Dict, marker_mapping: Dict,
                          line_mapping: Dict) -> List[Line2D]:
    color_handles = [Line2D([0], [0], color=color, lw=4, label=f"d={d}")
                     for d, color in color_mapping.items()]
    marker_handles = [Line2D([0], [0], color='black', marker=marker, lw=0,
                             markersize=10, label=f"k={k}")
                      for k, marker in marker_mapping.items()]
    line_handles = [Line2D([0], [0], color='black', linestyle=line, lw=1,
                           label=f"{dp}")
                    for dp, line in line_mapping.items()]

    return color_handles + marker_handles + line_handles


def plot_nicv_vs_key(data: Dict[str, pd.DataFrame], title: str, key: str, folder: str) -> None:
    plt.figure(figsize=FIGURE_SIZE)

    # Extract unique parameter values and create style mappings
    k_values = {extract_meta(dataset)[0] for dataset in data}
    d_values = {extract_meta(dataset)[1] for dataset in data}
    dps = data[list(data.keys())[0]]['dp'].unique()

    color_mapping, marker_mapping, line_mapping = create_style_mappings(k_values, d_values, dps)

    # Initialize data structures
    xs = {}
    ys = {}
    grouped_data = defaultdict(list)

    # Process each dataset
    for dataset in data:
        k, d = extract_meta(dataset)
        dataset_data, norm_value = normalize_data(data[dataset], dps)

        for dp in dps:
            exp_key = f"{k}_{d}_{dp}"
            exp_data = dataset_data[dataset_data['dp'] == dp]
            sorted_data = exp_data.sort_values(key)

            if len(sorted_data) == 0:
                continue

            xs[exp_key] = sorted_data[key].values.tolist()
            ys[exp_key] = (sorted_data[QUALITY_KEY].values / norm_value).tolist()
            grouped_data[exp_key].append(ys[exp_key])

    # Plot the data
    all_xs = np.unique(np.concatenate(list(xs.values())))

    # Plot individual experiments
    for group in grouped_data:
        k, d = map(int, group.split("_")[:2])
        dp = group.split("_")[-1]
        avg_ys = np.mean([group_data for group_data in grouped_data[group]], axis=0)
        plt.plot(all_xs, avg_ys, color=color_mapping[d], marker=marker_mapping[k],
                 linewidth=0.5, markersize=4, linestyle=line_mapping[dp], alpha=0.2)

    # Plot averages
    for dp in dps:
        # Overall average
        avg_ys = calculate_averages(xs, ys, all_xs, "", f"_{dp}")
        plt.plot(all_xs, avg_ys, linewidth=5, color='black',
                 linestyle=line_mapping[dp])

        # Averages by k
        for k in k_values:
            avg_ys = calculate_averages(xs, ys, all_xs, f"{k}_", f"_{dp}")
            plt.plot(all_xs, avg_ys, linewidth=2, color='black',
                     linestyle=line_mapping[dp], marker=marker_mapping[k])

        # Averages by d
        for d in d_values:
            avg_ys = calculate_averages(xs, ys, all_xs, "", f"{d}_{dp}")
            plt.plot(all_xs, avg_ys, linewidth=2, color=color_mapping[d],
                     linestyle=line_mapping[dp])

    # Customize plot
    plt.xlabel(key)
    plt.ylabel('NICV')
    plt.grid(True)

    # Add legend
    legend_handles = create_legend_handles(color_mapping, marker_mapping, line_mapping)
    plt.legend(handles=legend_handles, title="Legend", loc='center left',
               bbox_to_anchor=(1.05, 0.5), ncol=1)

    plt.tight_layout()

    # Save plot
    save_folder = f"figs/{folder}"
    os.makedirs(save_folder, exist_ok=True)
    plt.savefig(f"{save_folder}/{title}.png")
    plt.close()


def process_experiment_data(folder: str, key: str, param_name: str) -> None:
    per_eps = defaultdict(dict)

    # Read and organize data
    for dataset in exp_parameter_dict[f'{key}_ablation']['datasets']:
        file_path = f'{folder}/{key}_ablation/{dataset}/variances.csv'
        if not os.path.isfile(file_path):
            continue

        print(f"Processing: {file_path}")
        data = pd.read_csv(file_path)

        # Partition data by epsilon values
        for eps, eps_data in data.groupby('eps'):
            per_eps[eps][dataset] = eps_data

    # Generate plots for each epsilon value
    for eps, eps_data in per_eps.items():
        plot_nicv_vs_key(eps_data,
                         f"NICV vs {param_name} (eps={eps})",
                         key=param_name,
                         folder=folder)


def main():
    # Parameter mappings
    parameter_keys = {
        "maxdist": "alpha",
    }

    # Folders to process
    folders = [
        "chippie_new/constant_iters_04",
        "chippie_new/adapted_iters_04",
        # Add other folders as needed
    ]

    # Process each folder
    for folder in folders:
        print(f"\nProcessing folder: {folder}")
        for key, param_name in parameter_keys.items():
            try:
                process_experiment_data(folder, key, param_name)
                print(f"Successfully processed {param_name} parameter")
            except Exception as e:
                print(f"Error processing {param_name}: {str(e)}")


if __name__ == "__main__":
    main()
