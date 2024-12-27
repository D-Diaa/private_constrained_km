# Configs Package

This package manages configuration parameters and default settings for privacy-preserving federated clustering experiments.

## Overview

The configs package contains two main components:

1. `params.py`: Implements the `Params` class for managing clustering algorithm parameters, including:
   - Privacy settings (ε, δ, mechanism type)
   - Clustering parameters (number of clusters, dimensions)
   - Federated learning settings (number of clients, communication delay)
   - Constraint methods and post-processing options

2. `defaults.py`: Defines default experimental configurations, including:
   - Dataset selections for different experiment types
   - Privacy mechanism configurations
   - Parameter ranges for ablation studies
   - Dataset-specific settings (e.g., number of clusters)

## Usage

### Basic Parameter Configuration

```python
from configs import Params

# Create parameters with default values
params = Params()

# Create parameters with custom values
custom_params = Params(
    k=10,              # Number of clusters
    dim=2,             # Data dimensionality
    eps=1.0,           # Privacy budget
    dp="laplace",      # Privacy mechanism
    num_clients=5      # Number of federated clients
)
```

### Using Default Configurations

```python
from configs.defaults import exp_parameter_dict

# Get configurations for accuracy experiments
acc_configs = exp_parameter_dict["accuracy"]

# Get configurations for timing experiments
timing_configs = exp_parameter_dict["timing"]

# Access dataset-specific number of clusters
from configs.defaults import num_clusters
iris_clusters = num_clusters["iris"]  # Returns 3
```

## Experiment Types

The package supports several types of experiments:

1. **Accuracy Experiments**
   - Evaluate clustering quality under different privacy settings
   - Test various privacy budgets and mechanisms

2. **Timing Experiments**
   - Measure computational performance
   - Assess impact of communication delays

3. **Scaling Experiments**
   - Test with varying dataset sizes
   - Available in two variants: scale21 and scale70

4. **Ablation Studies**
   - Maximum distance constraint analysis
   - Post-processing method comparison

## Dataset Categories

The configurations include several dataset categories:

- Real-world benchmarks (iris, wine, mnist, etc.)
- Synthetic datasets with controlled parameters
- Low-dimensional datasets (2-15D)
- High-dimensional datasets (32-1024D)
- Specialized datasets for timing and scaling experiments