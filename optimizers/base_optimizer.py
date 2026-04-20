# -*- coding: cp1251 -*-
from abc import ABC, abstractmethod
import numpy as np
from typing import Any, List, Dict, Optional
from network.topology import TopologyManager


class BaseWorker(ABC):
    """
    Abstract class for a computational node.
    Encapsulates local memory (weights, gradient trackers) and step mathematics.
    """

    def __init__(
        self,
        worker_id: int,
        dim: int,
        lr: float,
        sigma: float,
        initial_state: Optional[Dict[str, np.ndarray]] = None,
    ):
        self.worker_id = worker_id
        self.dim = dim
        self.lr = lr
        self.sigma = sigma

        # Initialize state with provided data or default values
        if initial_state is not None:
            self.state = initial_state.copy()
            if "weights" not in self.state:
                self.state["weights"] = np.zeros(dim)
        else:
            self.state = {"weights": np.zeros(dim)}

    @abstractmethod
    def compute_local_step(
        self, local_grad: np.ndarray, **kwargs
    ) -> Dict[str, np.ndarray]:
        """Weight update mathematics based on the worker's local gradient. Must be implemented."""
        pass

    def update_state(self, new_state: Dict[str, np.ndarray]) -> None:
        """Update worker state after network exchange is complete."""
        self.state.update(new_state)


class BaseOptimizer(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def step(self, obs: Any) -> np.ndarray:
        pass

    @abstractmethod
    def reset(self) -> None:
        pass


class BaseDistributedOptimizer(BaseOptimizer, ABC):
    """
    Base class for simulating distributed algorithms.
    Runs sequentially, simulating discrete time ticks and data heterogeneity.
    """

    def __init__(
        self,
        dim: int,
        num_workers: int,
        lr: float,
        sigma: float = 0.01,
        topology: str = "all-to-all",
    ):
        super().__init__(name=self.__class__.__name__)
        self.dim = dim
        self.num_workers = num_workers
        self.lr = lr
        self.sigma = sigma

        # Topology matrix W
        self.topology_matrix = self._build_topology(topology)

        # Generate heterogeneity biases to simulate different local functions.
        # We ensure they sum to exactly 0 so the global minimum remains unchanged.
        # Fixed seed ensures the "local datasets" remain consistent across runs.
        rng = np.random.default_rng(42 + num_workers)
        actual_workers = getattr(self, "actual_workers_count", num_workers)
        self.heterogeneity_biases = rng.normal(0, 1.0, size=(actual_workers, dim))
        self.heterogeneity_biases -= np.mean(self.heterogeneity_biases, axis=0)

        # Create workers
        self.workers: List[BaseWorker] = [
            self._create_worker(i, dim, lr, sigma) for i in range(actual_workers)
        ]

    def _build_topology(self, topology: str) -> np.ndarray:
        """Generates the mixing matrix W."""
        actual_workers = getattr(self, "actual_workers_count", self.num_workers)
        return TopologyManager.generate_matrix(actual_workers, mode=topology)

    def _get_local_gradients(self, global_grad: np.ndarray) -> List[np.ndarray]:
        """
        Simulates heterogeneous local gradients.
        local_grad = global_grad + fixed_worker_bias + stochastic_noise
        """
        local_grads = []
        actual_workers = getattr(self, "actual_workers_count", self.num_workers)
        for i in range(actual_workers):
            noise = np.random.normal(0, self.sigma, size=global_grad.shape)
            local_grad = global_grad + self.heterogeneity_biases[i] + noise
            local_grads.append(local_grad)
        return local_grads

    @abstractmethod
    def _create_worker(
        self, worker_id: int, dim: int, lr: float, sigma: float
    ) -> BaseWorker:
        """Factory method for creating a specific type of worker."""
        pass

    @abstractmethod
    def _aggregate(
        self, states: List[Dict[str, np.ndarray]]
    ) -> List[Dict[str, np.ndarray]]:
        """Data exchange logic in the environment. Called at the end of each tick."""
        pass

    @abstractmethod
    def step(self, obs: Any) -> np.ndarray:
        """Step of the method."""
        pass

    def reset(self) -> None:
        actual_workers = getattr(self, "actual_workers_count", self.num_workers)
        self.workers = [
            self._create_worker(i, self.dim, self.lr, self.sigma)
            for i in range(actual_workers)
        ]
