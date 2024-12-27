from copy import copy

import numpy as np
from diffprivlib.mechanisms import Laplace, GaussianAnalytic

from data_io import to_fixed, to_int
from configs import Params


class Server:
    """Server implementation for privacy-preserving federated clustering.
    
    This class implements the server-side operations in a federated clustering system,
    with particular focus on privacy preservation through differential privacy mechanisms.
    The server aggregates masked statistics from clients and can apply differential
    privacy noise before sharing results back.
    
    The server supports two differential privacy mechanisms:
    1. Laplace mechanism: Provides ε-differential privacy
    2. Gaussian mechanism: Provides (ε,δ)-differential privacy
    
    The privacy budget (ε) is split between count queries and sum queries based on
    the configuration parameters.
    
    Attributes:
        count_mech: Differential privacy mechanism for count queries
        count_sensitivity (float): Sensitivity of count queries
        sum_mech: Differential privacy mechanism for sum queries
        sum_sensitivity (float): Sensitivity of sum queries
        params (Params): Configuration parameters for the clustering
        curr_iter (int): Current iteration number
        sum_eps (float): Privacy budget allocated for sum queries
        count_eps (float): Privacy budget allocated for count queries
    """

    def __init__(self, params: Params):
        """Initialize the server with given parameters.
        
        Args:
            params (Params): Configuration parameters for the clustering algorithm,
                           including privacy parameters and mechanism choices.
        """
        self.count_mech = None
        self.count_sensitivity = None

        self.sum_mech = None
        self.sum_sensitivity = None

        self.params = copy(params)
        self.curr_iter = 0
        self.sum_eps, self.count_eps = self.params.split_epsilon()

    def randomize_masked(self, sums, counts):
        """Add differential privacy noise to aggregated statistics.
        
        This method adds noise from the configured differential privacy mechanism
        (Laplace or Gaussian) to both the sum and count statistics. The noise
        is calibrated according to the sensitivity of the queries and the
        allocated privacy budget.
        
        Args:
            sums (np.ndarray): Aggregated sum statistics from all clients
            counts (np.ndarray): Aggregated count statistics from all clients
            
        Returns:
            tuple: A tuple containing:
                - np.ndarray: Noisy sum statistics
                - np.ndarray: Noisy count statistics
        """
        if self.params.dp != "none":
            for x in np.nditer(sums, op_flags=['readwrite']):
                noise = self.sum_mech.randomise(0)
                x[...] += to_fixed(noise)
            for x in np.nditer(counts, op_flags=['readwrite']):
                x[...] += to_int(self.count_mech.randomise(0))
        return sums, counts

    def dp_setup(self, params=None):
        """Configure the differential privacy mechanisms for the current iteration.
        
        Sets up either Laplace or Gaussian mechanisms based on the configuration,
        calculating appropriate sensitivities and allocating privacy budget.
        For Gaussian mechanism, delta is set based on data size for meaningful
        privacy guarantees.
        
        The sensitivity calculations take into account:
        - The dimensionality of the data
        - Whether bounded updates are enforced (max_dist parameter)
        - The impact of adding or removing a single data point
        
        Args:
            params (Params, optional): New parameters to update with. If None,
                                     uses the current parameters.
        """
        if params is None:
            params = self.params
        add_or_remove_delta = 1.0
        if self.params.dp == "laplace":
            # Calculate sensitivity for Laplace mechanism
            per_dim_sensitivity = add_or_remove_delta
            if params.method != "none":
                # If max_dist is enforced, then sensitivity could be bounded further
                per_dim_sensitivity = min(per_dim_sensitivity, params.max_dist)
            self.sum_sensitivity = per_dim_sensitivity * params.dim
            self.count_sensitivity = 1
            self.count_mech = Laplace(epsilon=self.count_eps,
                                      sensitivity=self.count_sensitivity,
                                      delta=0, random_state=params.seed * self.curr_iter)
            self.sum_mech = Laplace(epsilon=self.sum_eps,
                                    sensitivity=self.sum_sensitivity,
                                    delta=0, random_state=params.seed * self.curr_iter)
        elif "gaussian" in self.params.dp:
            # Calculate parameters for Gaussian mechanism
            delta = 1 / params.data_size ** 1.1
            # domain diagonal is the maximum possible distance between two points in the domain
            self.sum_sensitivity = np.sqrt(params.dim) * add_or_remove_delta
            if params.method != "none":
                # If max_dist is enforced, then sensitivity could be bounded further
                self.sum_sensitivity = min(self.sum_sensitivity, params.max_dist)
            self.count_sensitivity = 1
            self.sum_mech = GaussianAnalytic(epsilon=self.sum_eps,
                                             delta=delta, sensitivity=self.sum_sensitivity,
                                             random_state=params.seed * self.curr_iter)
            self.count_mech = GaussianAnalytic(epsilon=self.count_eps,
                                               delta=delta, sensitivity=self.count_sensitivity,
                                               random_state=params.seed * self.curr_iter)

    def step(self, masked_sums, masked_counts, params=None):
        """Perform one step of the federated clustering algorithm.
        
        This method:
        1. Sets up differential privacy mechanisms if enabled
        2. Aggregates masked statistics from all clients
        3. Applies differential privacy noise if enabled
        4. Advances the iteration counter
        
        Args:
            masked_sums (list): List of masked sum statistics from each client
            masked_counts (list): List of masked count statistics from each client
            params (Params, optional): New parameters to update with
            
        Returns:
            tuple: A tuple containing:
                - np.ndarray: Aggregated and possibly noisy sum statistics
                - np.ndarray: Aggregated and possibly noisy count statistics
        """
        # Update max_dist if constrained assignment and DP is enabled
        if self.params.dp != "none":
            self.dp_setup(params)
        masked_sum = sum(masked_sums)
        masked_count = sum(masked_counts)
        masked_sum, masked_count = self.randomize_masked(masked_sum, masked_count)
        self.curr_iter += 1
        return masked_sum, masked_count