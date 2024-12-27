"""
Module for generating heatmap visualizations of clustering algorithm scalability.

This module creates heatmap visualizations to analyze the scalability of different
clustering algorithms across varying numbers of clusters and dimensions. It supports
comparison between different methods (SuLloyd, Lloyd, FastLloyd, GLloyd) and can
generate both absolute performance heatmaps and relative performance comparisons.

The module handles multiple privacy settings (ε values) and can process both
adapted and constant iteration scenarios.
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.transforms import Bbox
from tqdm import tqdm

plt.rcParams.update({'font.size': 18})

quality_key = 'Normalized Intra-cluster Variance (NICV)'


# Refactored code
def extract_data(main_folder, method_names, eps=1):
    data_dict = {method: {} for method in method_names}

    for folder in tqdm(os.listdir(main_folder)):
        if folder.startswith("Synth"):
            for method in method_names:
                dist_method, dp = method
                if dp == "none":
                    _eps = 0
                else:
                    _eps = eps
                clusters, dimensions = map(int, folder.split('_')[1:3])
                variances_file_path = os.path.join(main_folder, folder, 'variances.csv')
                try:
                    df = pd.read_csv(variances_file_path)
                    # Using query method for cleaner filtering
                    query_str = f"dp == '{dp}' and method == '{dist_method}' and eps == {_eps}"
                    row = df.query(query_str)

                    if not row.empty and quality_key in row.columns:
                        nicv_value = row[quality_key].iloc[0]
                        if (clusters, dimensions) not in data_dict[method]:
                            data_dict[method][(clusters, dimensions)] = []
                        data_dict[method][(clusters, dimensions)].append(nicv_value)
                    else:
                        print(f"No 'laplace' data for {method} in file {variances_file_path}")
                except FileNotFoundError:
                    print(f"File not found: {variances_file_path}")
    for method in method_names:
        for key in data_dict[method]:
            vals = data_dict[method][key]
            data_dict[method][key] = sum(vals) / len(vals)

    return data_dict


def generate_heatmap_from_matrix(matrix, x_labels, y_labels, cmap, vmin, vmax, file_name, fmt=".2f", no_bar=False):
    plt.figure(figsize=(10, 8))
    if no_bar:
        sns.heatmap(matrix, annot=True, fmt=fmt, xticklabels=x_labels, yticklabels=y_labels, cmap=cmap, vmin=vmin,
                    vmax=vmax, cbar=False)
    else:
        sns.heatmap(matrix, annot=True, fmt=fmt, xticklabels=x_labels, yticklabels=y_labels, cmap=cmap, vmin=vmin,
                    vmax=vmax)
    plt.xlabel('Number of Dimensions')
    plt.ylabel('Number of Clusters')
    plt.gca().invert_yaxis()  # Invert the y-axis
    scale_exp = "70" if "70" in main_folder else "21"
    if not no_bar:
        bbox = Bbox([[7.5, -1], [9, 7.5]])
        plt.savefig(f'{main_folder}/{scale_exp}_{file_name}.png', bbox_inches=bbox)
    else:
        plt.savefig(f'{main_folder}/{scale_exp}_{file_name}.png')
    plt.close()  # Close the plot to prevent it from displaying in interactive environments


def create_heatmap(data, method_name, vmin, vmax, no_bar=False, eps=1):
    # Extract clusters and dimensions as separate lists
    clusters = sorted(set(key[0] for key in data.keys()))
    dimensions = sorted(set(key[1] for key in data.keys()))

    # Initialize a matrix to store NICV values
    nicv_matrix = np.zeros((len(clusters), len(dimensions)))

    # Populate the matrix with NICV values
    for i, cluster in enumerate(clusters):
        for j, dimension in enumerate(dimensions):
            nicv_matrix[i, j] = data.get((cluster, dimension), np.nan)  # Use NaN for missing values

    # Create a custom colormap: green (low), yellow (middle), red (high)
    cmap = sns.color_palette("RdYlGn_r", as_cmap=True)
    name = f'Heatmap_eps{eps}' + ("_bar" if not no_bar else f'_{method_name}')
    generate_heatmap_from_matrix(nicv_matrix, dimensions, clusters, cmap, vmin, vmax, name, no_bar=no_bar)

    return nicv_matrix


if __name__ == "__main__":
    method_names = {
        ("none", "laplace"): "SuLloyd",
        ("none", "none"): "Lloyd",
        # ("diagonal_then_adapted", "gaussiananalytic"): "FastLloyd",
        ("diagonal_then_constant", "gaussiananalytic"): "FastLloyd",
        ("none", "gaussiananalytic"): "GLloyd",
    }
    division_matrices = {
        "Fast vs SU": ("none", "laplace", "diagonal_then_constant", "gaussiananalytic"),
        # "Fast vs SU": ("none", "laplace", "diagonal_then_adapted", "gaussiananalytic"),
        "Fast vs G": ("none", "gaussiananalytic", "diagonal_then_constant", "gaussiananalytic"),
        # "Fast vs G": ("none", "gaussiananalytic", "diagonal_then_adapted", "gaussiananalytic"),
    }
    argc = len(sys.argv)
    if argc > 1:
        main_folder = sys.argv[1]
    else:
        main_folder = "chippie_new/constant_iters_04/scale70"
    folder = f"{main_folder}"
    print(f"Processing data in {folder}")
    for eps in [0.1, 0.5, 1, 2]:
        # Extract the data
        data_dict = extract_data(folder, method_names, eps)

        # Find the global min and max NICV values for the shared scale
        all_nicv_values = [value for method_data in data_dict.values() for value in method_data.values()]
        vmin, vmax = min(all_nicv_values), max(all_nicv_values)

        # Generate heatmaps for each methodregation method and store NICV matrices
        nicv_matrices = {}
        for method, data in data_dict.items():
            nicv_matrices[method] = create_heatmap(data, method, vmin, vmax, no_bar=True, eps=eps)

        create_heatmap(data, method, vmin, vmax, no_bar=False)
        # Calculate the division matrix (ensure to handle division by zero appropriately)
        for name, keys in division_matrices.items():
            first_matrix = nicv_matrices[keys[:2]]
            second_matrix = nicv_matrices[keys[2:]]
            division_matrix = np.divide(first_matrix - second_matrix, first_matrix, out=np.zeros_like(first_matrix),
                                        where=second_matrix != 0)
            # Assuming division_matrix, and the labels (dimensions and clusters) are already defined
            # cmap_division = sns.color_palette("YlGn", as_cmap=True)
            # cmao with shades of blue
            cmap_division = sns.color_palette("Blues", as_cmap=True)
            vmin_div, vmax_div = np.nanmin(division_matrix), np.nanmax(division_matrix)

            # Generate the heatmap for the division matrix
            generate_heatmap_from_matrix(division_matrix, sorted(set(key[1] for key in data.keys())),
                                         sorted(set(key[0] for key in data.keys())), cmap_division, vmin_div, vmax_div,
                                         f'Heatmap_eps{eps}_{name}', fmt=".0%", no_bar=True)
