"""
Module for creating bar plots comparing clustering algorithm performance on synthetic datasets.

This module processes and visualizes the performance of different clustering methods
on synthetic datasets, focusing on comparing their Area Under Curve (AUC) metrics
across different dimensions and dataset families. It supports multiple dataset
families (e.g., 'g2', 'dim') and normalizes results relative to SuLloyd's performance.
"""

import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import argparse


def load_data(data_dir, dataset_family):
    """Load and preprocess data from the specified directory and dataset family.

    Args:
        data_dir (str): Path to the data directory.
        dataset_family (str): Dataset family prefix (e.g., 'g2', 'dim').

    Returns:
        dict: A dictionary with dimensions as keys and AUC values per method as values.
    """
    # Define a standard range of epsilon values for interpolation
    standard_epsilons = np.linspace(0, 2, 100)

    # Define the mapping for method names based on "method" and "dp" columns
    method_names = {
        ("none", "laplace"): "SuLloyd",
        ("none", "none"): "Lloyd",
        ("diagonal_then_adapted", "gaussiananalytic"): "FastLloyd",
        ("diagonal_then_constant", "gaussiananalytic"): "FastLloyd",
        ("none", "gaussiananalytic"): "GLloyd",
    }

    dataset_folders = [folder for folder in os.listdir(data_dir) if folder.startswith(dataset_family)]

    # Initialize dictionary to store aggregated data
    dimension_data_auc = {}

    for folder in dataset_folders:
        folder_path = os.path.join(data_dir, folder)
        csv_path = os.path.join(folder_path, 'variances.csv')

        # Check if variances.csv exists
        if os.path.exists(csv_path):
            # Read the CSV file
            data = pd.read_csv(csv_path)

            # Extract the dimension from the folder name
            if dataset_family == 'g2':
                dimension = int(folder.split('-')[1])
            elif dataset_family == 'dim':
                dimension = int(folder.replace('dim', ''))
            else:
                raise ValueError(f"Unknown dataset family: {dataset_family}")

            # Aggregate NICV values by method and dimension
            if dimension not in dimension_data_auc:
                dimension_data_auc[dimension] = {}

            for (method, dp), group in data.groupby(['method', 'dp']):
                if (method, dp) in method_names:
                    method_name = method_names[(method, dp)]
                    epsilons = group['eps'].values
                    sorted_indices = np.argsort(epsilons)
                    epsilons = epsilons[sorted_indices]
                    nicvs = group['Normalized Intra-cluster Variance (NICV)'].values
                    nicvs = nicvs[sorted_indices]

                    # Handle cases with only one point
                    if len(epsilons) == 1:
                        epsilons = standard_epsilons
                        nicvs = np.full_like(standard_epsilons, nicvs[0])

                    # Interpolate NICV values over the standard range of epsilons
                    interpolated_nicvs = np.interp(standard_epsilons, epsilons, nicvs)

                    # Calculate the AUC for the method and dimension
                    auc = np.trapz(interpolated_nicvs, x=standard_epsilons)
                    if method_name not in dimension_data_auc[dimension]:
                        dimension_data_auc[dimension][method_name] = 0
                    # For dimensions with multiple datasets, sum the AUC values
                    dimension_data_auc[dimension][method_name] += auc

    # Normalize AUC values by SuLloyd
    for dimension, methods in dimension_data_auc.items():
        sulloyd_auc = methods.get("SuLloyd", 1)  # Avoid division by zero
        for method in methods:
            methods[method] /= sulloyd_auc

    return dimension_data_auc


def plot_data(dimension_data_auc, dataset_family, data_dir):
    """Plot the normalized AUC data.

    Args:
        dimension_data_auc (dict): Aggregated data for plotting.
        dataset_family (str): Dataset family name for labeling.
        data_dir (str): Directory to save the plots.
    """
    # Convert to DataFrame for easier plotting
    df_auc = pd.DataFrame(dimension_data_auc).T

    # Sort by dimension
    df_auc.sort_index(inplace=True)

    # Plotting
    plt.figure(figsize=(10, 6))
    methods_auc = df_auc.columns
    x = np.arange(len(df_auc.index))  # Number of dimensions

    # Plot bars for each method
    bar_width = 0.2
    for i, method in enumerate(methods_auc):
        plt.bar(x + i * bar_width, df_auc[method], width=bar_width, label=method)

    # Formatting
    plt.xlabel("Number of Dimensions")
    plt.ylabel("Normalized AUC (Relative to SuLloyd)")
    plt.title(f"Normalized AUC of NICV per Method for Different Dimensions ({dataset_family} Datasets)")
    plt.xticks(x + bar_width * (len(methods_auc) - 1) / 2, df_auc.index)
    plt.legend(title="Method")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    # Save the plot
    output_path = os.path.join(data_dir, f"{dataset_family}_auc.png")
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")


def main():
    """Main function to execute the data loading, processing, and plotting."""
    parser = argparse.ArgumentParser(description="Process NICV data and plot AUC.")
    parser.add_argument("--data_dir", type=str, default="chippie_new/adapted_iters_04/accuracy",
                        help="Path to the directory containing dataset folders.")
    args = parser.parse_args()

    for dataset_family in ["g2", "dim"]:
        dimension_data_auc = load_data(args.data_dir, dataset_family)
        plot_data(dimension_data_auc, dataset_family, args.data_dir)


if __name__ == "__main__":
    main()
