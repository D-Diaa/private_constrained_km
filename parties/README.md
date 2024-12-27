# Privacy-Preserving Federated Clustering

The `parties` package implements a privacy-preserving federated clustering system where multiple clients can
collaboratively learn cluster centroids without sharing their raw data. This implementation supports both standard
federated learning and privacy-enhanced versions using masking and differential privacy.

## Core Components

### Client Implementation

The package provides two types of clients through the `client.py` module:

1. `UnmaskedClient`: Implements standard federated clustering where clients compute and share statistics directly. This
   is suitable for scenarios where privacy is not a primary concern.

2. `MaskedClient`: Extends the base client to add privacy-preserving features through masking. Before sharing statistics
   with the server, each client adds random masks that sum to zero across all clients, ensuring that individual
   contributions remain private.

Both client types support bounded updates through the `max_dist` parameter, which helps prevent large changes to
centroids and can improve convergence.

### Server Implementation

The `server.py` module implements the central server that coordinates the federated clustering process. Key features
include:

1. Aggregation of client statistics (masked or unmasked)
2. Differential privacy support through:
    - Laplace mechanism for ε-differential privacy
    - Gaussian mechanism for (ε,δ)-differential privacy
3. Dynamic privacy budget allocation between count and sum queries

## Privacy Guarantees

The system provides privacy through multiple complementary mechanisms:

1. Masking: Clients never share raw statistics, only masked versions
2. Differential Privacy: The server adds calibrated noise to aggregated statistics
3. Bounded Updates: Limiting the maximum change in centroids helps bound sensitivity
4. Split Privacy Budget: The privacy budget is carefully divided between different types of queries

## Usage Example

Here's a simple example of setting up a privacy-preserving federated clustering system:

```python
from parties import MaskedClient, Server
from configs import Params

# Configure the clustering parameters
params = Params(
    k=3,  # Number of clusters
    dim=2,  # Data dimensionality
    max_dist=0.1,  # Maximum centroid movement
    dp="gaussian",  # Differential privacy mechanism
    epsilon=1.0,  # Privacy budget
    num_clients=5  # Number of participating clients
)

# Create clients with their local data
clients = [
    MaskedClient(
        index=i,
        values=client_data[i],
        params=params
    ) for i in range(params.num_clients)
]

# Create the server
server = Server(params)

# Run the federated clustering algorithm
for iteration in range(params.iters):
    # Client-side computations
    totals = []
    counts = []
    for client in clients:
        total, count, _ = client.step(params)
        totals.append(total)
        counts.append(count)

    # Server-side aggregation with privacy
    total, count = server.step(totals, counts, params)

    # Update clients
    for client in clients:
        client.update(total, count)
```