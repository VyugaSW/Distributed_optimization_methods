# -*- coding: cp1251 -*-
import numpy as np
from typing import List, Optional, Tuple


class TopologyManager:
    """
    Class for generating network topologies.
    Creates a matrix W of size (n x n), where W[i][j] is the connection weight between nodes i and j.
    """

    @staticmethod
    def generate_matrix(n: int, mode: str = "all-to-all") -> np.ndarray:
        """
        Main generation method.
        Implements the doubly stochastic matrix condition (sum of rows and columns = 1).
        """
        if n <= 1:
            return np.array([[1.0]])

        if mode == "all-to-all":
            # Complete graph (star): each node connected to every other
            return np.full((n, n), 1.0 / n)

        elif mode == "ring":
            # Ring: node i connected to i-1 and i+1
            W = np.zeros((n, n))
            for i in range(n):
                W[i, i] = 1 / 3
                W[i, (i - 1) % n] = 1 / 3
                W[i, (i + 1) % n] = 1 / 3
            return W

        elif mode == "star":
            # Star: node 0 is the central hub
            W = np.zeros((n, n))
            W[0, :] = 1.0 / n
            for i in range(1, n):
                W[i, 0] = 0.5
                W[i, i] = 0.5
            return W

        elif mode == "directed_ring":
            W = np.zeros((n, n))
            for i in range(n):
                W[i, i] = 0.5
                W[(i + 1) % n, i] = 0.5
            return W

        elif mode == "grid":
            side = int(np.sqrt(n))
            if side * side != n:
                return TopologyManager.generate_matrix(n, "ring")

            W = np.eye(n)
            for i in range(n):
                neighbors = []
                if i % side > 0:
                    neighbors.append(i - 1)  # left
                if i % side < side - 1:
                    neighbors.append(i + 1)  # right
                if i >= side:
                    neighbors.append(i - side)  # up
                if i < n - side:
                    neighbors.append(i + side)  # down

                deg = len(neighbors)
                W[i, i] = 1.0 / (deg + 1)
                for nb in neighbors:
                    W[i, nb] = 1.0 / (deg + 1)
            return W

        else:
            raise ValueError(f"Topology '{mode}' is not supported.")

    @staticmethod
    def generate_push_pull_matrices(
        n: int, mode: str = "directed_ring"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generates a pair of matrices R and C specifically for the Push-Pull algorithm.
        R: Row-stochastic (Pull matrix for weights X, sum of rows = 1)
        C: Column-stochastic (Push matrix for gradients Y, sum of columns = 1)
        """
        if n <= 1:
            return np.array([[1.0]]), np.array([[1.0]])

        R = np.zeros((n, n))
        C = np.zeros((n, n))

        if mode == "directed_ring" or mode == "ring":
            # Directed ring: information flows only clockwise
            for i in range(n):
                # R: row i sums to 1
                R[i, i] = 0.5
                R[i, (i - 1) % n] = 0.5

                # C: column i sums to 1
                C[i, i] = 0.5
                C[(i + 1) % n, i] = 0.5

        elif mode == "all-to-all":
            R = np.full((n, n), 1.0 / n)
            C = np.full((n, n), 1.0 / n)

        elif mode == "undirected_ring":
            W = TopologyManager.generate_matrix(n, "ring")
            R, C = W.copy(), W.copy()

        else:
            raise ValueError(f"Directed topology '{mode}' is not supported.")
        return R, C

    @staticmethod
    def get_spectral_gap(W: np.ndarray) -> float:
        """
        Metric: spectral gap.
        """
        eigvals = np.sort(np.abs(np.linalg.eigvals(W)))
        return 1 - eigvals[-2]
