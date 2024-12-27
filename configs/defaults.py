"""Default configuration parameters for federated clustering experiments.

This module defines the default parameters and dataset configurations for various
experimental setups in privacy-preserving federated clustering. It includes settings
for different types of experiments:
- Timing experiments (measuring computational performance)
- Accuracy experiments (measuring clustering quality)
- Scaling experiments (testing with different dataset sizes)
- Ablation studies (analyzing impact of specific parameters)

The configurations cover:
- Dataset selections
- Privacy mechanisms and parameters
- Experimental variables
- Cluster counts for different datasets
"""

import os

import numpy as np

# Dataset configurations for ablation studies
ablate_dataset = [
    f"AblateSynth_{k}_{d}_{sep}"
    for k in [2, 4, 8, 16]  # Number of clusters
    for d in [2, 4, 8, 16]  # Dimensions
    for sep in [0.25, 0.5, 0.75]  # Cluster separation
]

# Dataset configurations for dimensionality experiments
dim_low = [f"dim{d}" for d in range(2, 16)]  # Low-dimensional datasets (2-15D)
dim_high = [f"dim{d}" for d in [32, 64, 128, 256, 512, 1024]]  # High-dimensional datasets

# G2 datasets (gathered from data directory)
g2 = [file.replace(".txt", "") for file in os.listdir("data") if file.startswith("g2")]

# Real-world benchmark datasets
real_datasets = ["iris", "s1", "house", "adult", "lsun", "birch2", "wine", "yeast", "breast", "mnist"]

# Combined datasets for accuracy experiments
accuracy_datasets = dim_low + dim_high + g2 + real_datasets

# Datasets for timing experiments
timing_datasets = [
                      "s1", "lsun"  # Real datasets
                  ] + [
                      # Synthetic datasets with varying parameters
                      f"timesynth_{k}_{d}_{n}"
                      for k in [2, 5]  # Number of clusters
                      for d in [2, 5]  # Dimensions
                      for n in [10000, 100000]  # Number of points
                  ]

# Datasets for scaling experiments
scale21_datasets = [file.replace(".txt", "") for file in os.listdir("data") if file.startswith("Synth21")]
scale70_datasets = [file.replace(".txt", "") for file in os.listdir("data") if file.startswith("Synth70")]

# Experimental configurations

timing_parameters = {
    "dps": ["gaussiananalytic"],  # Privacy mechanisms
    "eps_budgets": [1],  # Privacy budgets
    "delays": [0.000125, 0.025],  # Communication delays
    "datasets": timing_datasets,  # Datasets to use
}

acc_parameters = {
    "dps": ["none", "laplace", "gaussiananalytic"],  # Privacy mechanisms
    "methods": ["none", "diagonal_then_adapted"],  # Constraint methods
    "eps_budgets": [0.1, 0.25, 0.5, 0.75, 1, 1.5, 2.0, 3.0, 4.0],  # Privacy budgets
    "delays": [0],  # Communication delays
    "datasets": accuracy_datasets,  # Datasets to use
}

scale21_parameters = {
    "dps": ["none", "laplace", "gaussiananalytic"],  # Privacy mechanisms
    "methods": ["none", "diagonal_then_adapted"],  # Constraint methods
    "eps_budgets": [0.1, 0.5, 1, 2],  # Privacy budgets
    "delays": [0],  # Communication delays
    "datasets": scale21_datasets,  # Datasets to use
}

# Copy scale21 parameters and update datasets for scale70
scale70_parameters = scale21_parameters.copy()
scale70_parameters["datasets"] = scale70_datasets

max_dist_ablation_parameters = {
    "dps": ["none", "laplace", "gaussiananalytic"],  # Privacy mechanisms
    "methods": ["none", "diagonal_then_adapted"],  # Constraint methods
    "eps_budgets": [0.1, 0.5, 1],  # Privacy budgets
    "posts": ["fold"],  # Post-processing methods
    "delays": [0],  # Communication delays
    "datasets": ablate_dataset,  # Datasets to use
    "alphas": np.linspace(0.7, 1.2, 50),  # Range of alpha values to test
}

post_ablation_parameters = {
    "dps": ["none", "laplace", "gaussiananalytic"],  # Privacy mechanisms
    "eps_budgets": [0.1, 1],  # Privacy budgets
    "posts": ["none", "fold", "truncate"],  # Post-processing methods
    "delays": [0],  # Communication delays
    "datasets": ablate_dataset,  # Datasets to use
}

# Number of clusters for each dataset
num_clusters = {
    # Real-world datasets
    "iris": 3,
    "s1": 15,
    "birch2": 100,
    "house": 3,
    "adult": 3,
    "lsun": 3,
    "wine": 3,
    "yeast": 10,
    "breast": 2,
    "mnist": 10,
}

# Update num_clusters for dimensional datasets
num_clusters.update({
    dataset: 9 for dataset in dim_low  # Low-dimensional datasets
})

num_clusters.update({
    dataset: 16 for dataset in dim_high  # High-dimensional datasets
})

num_clusters.update({
    dataset: 2 for dataset in g2  # G2 datasets
})

# Dictionary mapping experiment types to their parameters
exp_parameter_dict = {
    "timing": timing_parameters,  # Timing experiments
    "accuracy": acc_parameters,  # Accuracy experiments
    "post_ablation": post_ablation_parameters,  # Post-processing ablation study
    "maxdist_ablation": max_dist_ablation_parameters,  # Maximum distance ablation study
    "scale21": scale21_parameters,  # Scaling experiments (21 separation parameter)
    "scale70": scale70_parameters,  # Scaling experiments (70 separation parameter)
}
