from copy import copy
from numbers import Real

import numpy as np

from data_io import MOD, to_fixed, unscale, SCALE
from configs import Params
from utils import distance_matrix_squared


class UnmaskedClient:
    """A client implementation for federated clustering without masking/encryption.
    
    This class implements the client-side operations for federated k-means clustering,
    including local data processing, centroid updates, and bounded updates when using
    constraint-based methods.
    
    Attributes:
        params (Params): Configuration parameters for the clustering algorithm
        associations (np.ndarray): Cluster assignments for each data point
        index (int): Unique identifier for this client
        curr_iter (int): Current iteration number in the clustering process
        post (TruncationAndFolding): Post-processing object for value bounds
        values (np.ndarray): Local data points owned by this client
        centroids (np.ndarray): Current cluster centroids
    """

    def __init__(self, index: int, values, params: Params):
        """Initialize a new unmasked client.
        
        Args:
            index (int): Unique identifier for this client
            values (np.ndarray): Local data points owned by this client
            params (Params): Configuration parameters for the clustering
        
        Raises:
            AssertionError: If the dimensionality of values doesn't match params.dim
        """
        self.params = copy(params)
        self.associations = None
        self.index = index
        self.curr_iter = 0
        self.post = TruncationAndFolding(-1, 1)

        assert values.shape[1] == params.dim, "Error! values shape doesn't match"
        if params.fixed:
            values = unscale(values)
        self.values = values
        self.centroids = self.params.init_centroids()

    def update(self, totals, counts):
        """Update local centroids based on aggregated totals and counts.
        
        Updates the cluster centroids using the aggregated totals and counts from all clients,
        applying bounds and post-processing if specified in the parameters.
        
        Args:
            totals (np.ndarray): Sum of all data points per cluster across all clients
            counts (np.ndarray): Number of points per cluster across all clients
        """
        if self.params.fixed:
            totals = unscale(totals)
        # correct the shift
        if self.params.method != "none":
            totals += self.centroids * np.expand_dims(counts, axis=1)

        new_centroids = totals / (np.expand_dims(counts, axis=1) + 1e-6)
        if self.params.method != "none":
            centroid_diff = new_centroids - self.centroids
            distance = np.linalg.norm(centroid_diff, axis=1)
            mask = distance > self.params.max_dist
            direction = centroid_diff / distance[:, None]
            new_centroids[mask] = self.centroids[mask] + self.params.max_dist * direction[mask]

        self.centroids = new_centroids
        if self.params.post == "fold":
            self.centroids = self.post.fold(self.centroids)
        elif self.params.post == "truncate":
            self.centroids = self.post.truncate(self.centroids)
        elif np.any(self.centroids < -1) or np.any(self.centroids > 1):
            print("Warning: Constrained Centroids are outside the bounds")
        self.curr_iter += 1

    def local_step(self, old_centroids):
        """Compute local statistics for each cluster.
        
        Calculates the sum of points and count of points per cluster based on current
        cluster assignments.
        
        Args:
            old_centroids (np.ndarray): Previous cluster centroids
            
        Returns:
            tuple: A tuple containing:
                - np.ndarray: Sum of points per cluster
                - np.ndarray: Count of points per cluster
        """
        local_totals = []
        local_counts = []
        for cluster in range(self.params.k):
            indices = self.associations == cluster
            count = sum(indices)
            if count > 0:
                local_total = self.values[indices].sum(axis=0)
            else:
                count = 1
                local_total = old_centroids[cluster]
            local_counts.append(count)
            local_totals.append(local_total)
        local_totals = np.array(local_totals)
        local_counts = np.array(local_counts, dtype=np.int32)
        return local_totals, local_counts

    def step(self, params=None):
        """Perform one step of the federated clustering algorithm.
        
        Updates cluster assignments for local data points and computes necessary
        statistics for global aggregation.
        
        Args:
            params (Params, optional): New parameters to update with
            
        Returns:
            tuple: A tuple containing:
                - np.ndarray: Sum of points per cluster
                - np.ndarray: Count of points per cluster
                - int: Number of points not assigned to any cluster
        """
        squared_distances = distance_matrix_squared(self.values, self.centroids)
        self.associations = np.argmin(squared_distances, axis=1)
        if params is not None:
            self.params = params
        squared_max_dist = self.params.max_dist ** 2
        self.associations[squared_distances.min(axis=1) > squared_max_dist] = params.k
        local_totals, local_counts = self.local_step(self.centroids)
        if self.params.method != "none":
            # shift the sums to the origin
            local_totals -= self.centroids * np.expand_dims(local_counts, axis=1)
        if self.params.fixed:
            local_totals = to_fixed(local_totals)
        unassigned_count = sum(self.associations == self.params.k)
        return local_totals, local_counts, unassigned_count


class MaskedClient(UnmaskedClient):
    """A client implementation that uses masking for privacy-preserving federated clustering.
    
    Extends UnmaskedClient to add privacy-preserving features through masking of local
    statistics before sharing with the server. Uses a secure multi-party computation
    approach where masks sum to zero across all clients.
    
    Additional Attributes:
        sum_dmasks (np.ndarray): Differential masks for sums across all iterations
        sum_emasks (np.ndarray): Local masks for sums for this client
        count_dmasks (np.ndarray): Differential masks for counts across all iterations
        count_emasks (np.ndarray): Local masks for counts for this client
    """

    def __init__(self, index: int, values, params: Params):
        """Initialize a new masked client.
        
        Args:
            index (int): Unique identifier for this client
            values (np.ndarray): Local data points owned by this client
            params (Params): Configuration parameters for the clustering
        """
        super().__init__(index, values, params)
        generator = np.random.RandomState(params.seed)
        sum_masks = [
            np.array([generator.randint(0, MOD, (params.k, params.dim))
                      for _ in range(params.iters + 1)])
            for __ in range(params.num_clients)
        ]
        self.sum_dmasks: np.ndarray = np.sum(sum_masks, axis=0)
        self.sum_emasks = sum_masks[index]
        count_masks = [
            np.array([generator.randint(0, MOD, params.k)
                      for _ in range(params.iters + 1)])
            for __ in range(params.num_clients)
        ]
        self.count_dmasks: np.ndarray = np.sum(count_masks, axis=0)
        self.count_emasks = count_masks[index]

    def step(self, params=None):
        """Perform one step of the privacy-preserving federated clustering algorithm.
        
        Extends the parent class step method by adding masks to the local statistics
        before sharing.
        
        Args:
            params (Params, optional): New parameters to update with
            
        Returns:
            tuple: A tuple containing:
                - np.ndarray: Masked sum of points per cluster
                - np.ndarray: Masked count of points per cluster
                - int: Number of points not assigned to any cluster
        """
        totals, counts, unassigned = super().step(params)
        masked_totals = self.sum_emasks[self.curr_iter] + totals
        masked_counts = self.count_emasks[self.curr_iter] + counts
        return masked_totals, masked_counts, unassigned

    def update(self, masked_totals, masked_counts):
        """Update local centroids using masked aggregated statistics.
        
        Removes the masks from the aggregated statistics before updating centroids.
        
        Args:
            masked_totals (np.ndarray): Masked sum of all points per cluster
            masked_counts (np.ndarray): Masked count of points per cluster
        """
        totals = masked_totals - self.sum_dmasks[self.curr_iter]
        counts = masked_counts - self.count_dmasks[self.curr_iter]
        super().update(totals, counts)


class TruncationAndFolding:
    """Implements truncation and folding operations for bounded data.
    
    This class provides methods to ensure data stays within specified bounds either
    through truncation (clipping) or folding (reflection about boundaries).
    
    Originally from: https://github.com/IBM/differential-privacy-library
    
    Attributes:
        lower (float): Lower bound for the data
        upper (float): Upper bound for the data
        lower_f (float): Scaled lower bound for fixed-point arithmetic
        upper_f (float): Scaled upper bound for fixed-point arithmetic
    """

    def __init__(self, lower, upper):
        """Initialize the bounds for truncation and folding.
        
        Args:
            lower (float): Lower bound for the data
            upper (float): Upper bound for the data
            
        Raises:
            TypeError: If bounds are not numeric
            ValueError: If lower bound is greater than upper bound
        """
        self.lower, self.upper = self._check_bounds(lower, upper)
        self.lower_f, self.upper_f = self.lower * SCALE, self.upper * SCALE

    @classmethod
    def _check_bounds(cls, lower, upper):
        """Validates the provided bounds.
        
        Args:
            lower (float): Lower bound to check
            upper (float): Upper bound to check
            
        Returns:
            tuple: Validated (lower, upper) bounds
            
        Raises:
            TypeError: If bounds are not numeric
            ValueError: If lower bound is greater than upper bound
        """
        if not isinstance(lower, Real) or not isinstance(upper, Real):
            raise TypeError("Bounds must be numeric")

        if lower > upper:
            raise ValueError("Lower bound must not be greater than upper bound")

        return lower, upper

    def _truncate(self, value):
        """Clip a single value to the specified bounds.
        
        Args:
            value (float): Value to truncate
            
        Returns:
            float: Truncated value within bounds
        """
        if value > self.upper:
            return self.upper
        if value < self.lower:
            return self.lower

        return value

    def _fold(self, value):
        """Fold a single value about the bounds until it lies within them.
        
        Args:
            value (float): Value to fold
            
        Returns:
            float: Folded value within bounds
        """
        while value < self.lower_f or value > self.upper_f:
            if value < self.lower_f:
                value = (2 * self.lower_f - value)
            else:
                value = (2 * self.upper_f - value)
        return value

    def fold(self, values):
        """Apply folding to an array of values.
        
        Args:
            values (np.ndarray): Array of values to fold
            
        Returns:
            np.ndarray: Array with all values folded within bounds
        """
        fixed_vals = to_fixed(values)
        for x in np.nditer(fixed_vals, op_flags=['readwrite']):
            x[...] = self._fold(x[...])
        return unscale(fixed_vals)

    def truncate(self, values):
        """Apply truncation to an array of values.
        
        Args:
            values (np.ndarray): Array of values to truncate
            
        Returns:
            np.ndarray: Array with all values truncated within bounds
        """
        for x in np.nditer(values, op_flags=['readwrite']):
            x[...] = self._truncate(x[...])
        return values