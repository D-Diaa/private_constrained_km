# Clustering Utilities Package

## Package Structure

The package consists of several modules that work together to support clustering operations:

### protocols.py

This module implements the core protocols for federated clustering, enabling distributed computation across multiple
parties while maintaining privacy guarantees. It provides:

1. MPI-based Protocol (`mpi_proto`)
    - Enables real distributed computation using Message Passing Interface (MPI)
    - Supports both masked (privacy-preserving) and unmasked operations
    - Handles communication between multiple client processes and a central server
    - Includes built-in network delay simulation and communication statistics tracking

2. Local Protocol (`local_proto`)
    - Simulates federated clustering in a single process
    - Useful for testing, development, and algorithm validation
    - Maintains identical functionality to MPI protocol but with simplified communication
    - Tracks clustering progress through centroid movement

Both protocols support:

- Privacy-preserving computation through masking techniques
- Configurable parameters for clustering behavior
- Progress tracking and monitoring
- Detailed error handling and reporting

### evaluations.py

This module implements various metrics to evaluate clustering quality. It includes:

- Normalized Intra-cluster Variance (NICV): Measures the average variance within clusters, helping assess cluster
  compactness
- Between-Cluster Sum of Squares (BCSS): Quantifies separation between clusters
- Empty cluster detection: Identifies clusters with no assigned points

### utils.py

Contains core utility functions used throughout the clustering implementation, such as:

- Distance calculations between points and centroids
- Matrix operations optimized for clustering computations
- Helper functions for data preprocessing and manipulation

## Usage Examples

### Using the Federated Clustering Protocols

Here's an example of how to use the local protocol for testing:

```python
import numpy as np
from utils.protocols import local_proto
from configs import Params

# Prepare your data
client_data = [
    np.array([[1, 1], [2, 2]]),  # Client 1's data
    np.array([[4, 4], [5, 5]])  # Client 2's data
]

# Configure clustering parameters
params = Params(
    k=2,  # Number of clusters
    dim=2,  # Data dimensionality
    num_clients=2,  # Number of participating clients
    iters=10,  # Number of iterations
    seed=42  # Random seed for reproducibility
)

# Run the clustering protocol
centroids, unassigned = local_proto(
    value_lists=client_data,
    params=params,
    method="masked"  # Use privacy-preserving computation
)

print("Final centroids:", centroids)
print("Unassigned points:", unassigned)
```

### Evaluating Clustering Results

```python
from utils.evaluations import evaluate

# Evaluate clustering quality
metrics = evaluate(centroids, client_data[0])  # Evaluate on first client's data

print(f"NICV: {metrics['Normalized Intra-cluster Variance (NICV)']}")
print(f"BCSS: {metrics['Between-Cluster Sum of Squares (BCSS)']}")
print(f"Empty Clusters: {metrics['Empty Clusters']}")
```
