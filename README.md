# Privacy-Preserving Federated Clustering

A framework for performing federated k-means clustering with differential privacy guarantees.
## Features
- Federated k-means clustering implementation with configurable client count
- Multiple privacy preservation mechanisms:
  - Laplace mechanism (ε-differential privacy)
  - Analytic Gaussian mechanism ((ε,δ)-differential privacy)
  - Fixed-point arithmetic for secure computations
- Flexible experiment framework supporting:
  - Multiple datasets (including MNIST and synthetic data)
  - Configurable network delays (LAN/WAN simulation)
  - Parallel experiment execution (for accuracy, scaling, etc)
  - Comprehensive metrics collection and visualization
  - MPI-based distributed computing

## Installation

### Environment Setup
The project uses conda for dependency management. Create the environment using:

```bash
conda env create -f env.yml
conda activate privatekm
```

Main dependencies:
- Python 3.11.5
- NumPy, Pandas, Scikit-learn
- mpi4py 3.1.4 (for distributed computing)
- diffprivlib 0.6.2 (for differential privacy mechanisms)
- Matplotlib, Seaborn (for visualization)

## Usage

### Running Experiments

Basic experiment execution:
```bash
python experiments.py --datasets mnist --exp_type testing
```

Key command line arguments:
- `--exp_type`: Type of experiment (default: "accuracy")
- `--datasets`: List of datasets to process
- `--plot`: Enable cluster visualization
- `--num_runs`: Number of experiment repetitions (default: 10)
- `--method`: Distance calculation method ("none" or "diagonal_then_adapted")
- `--post`: Centroid post-processing method ("none", "truncate", or "fold")
- `--results_folder`: Output directory for results

### Distributed Execution
For MPI-based distributed experiments:
```bash
mpirun -n <num_processes> python experiments.py --exp_type timing
```

## Project Structure
- `data_io/`: Data handling and communication
  - `comm.py`: MPI communication utilities
  - `data_handler.py`: Dataset loading and preprocessing
  - `fixed.py`: Fixed-point arithmetic implementation
- `utils/`: Core functionality
  - Evaluation metrics
  - Clustering protocols
  - Visualization tools
- `configs/`: Configuration parameters and experiment settings
- `parties/`: Implementation of different party roles (server/clients)
- `plots/`: Output directory for visualizations
- `experiments.py`: Main experiment runner

## Evaluation Metrics
- Normalized Inter-Cluster Variance (NICV)
- Within-Cluster Sum of Squares (WCSS)
- Between-Cluster Sum of Squares (BCSS)
- Empty cluster count
- Execution time
- Communication overhead (for distributed experiments)

## Privacy Mechanisms
The framework implements multiple privacy-preserving techniques:
- Differential Privacy:
  - Laplace mechanism with configurable ε
  - Analytic Gaussian mechanism with ε, δ
- Secure Computation:
  - Fixed-point arithmetic
  - Configurable post-processing methods (truncation, folding)

## Custom Datasets
Custom datasets can be added as text files in the `data/` directory. The framework supports:
- Tabular data in text format
- Automatic data normalization
- Fixed-point conversion for secure computation
