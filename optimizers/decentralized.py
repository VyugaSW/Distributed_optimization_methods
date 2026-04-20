# -*- coding: cp1251 -*-
from optimizers.base_optimizer import BaseWorker, BaseDistributedOptimizer
from network.topology import TopologyManager
from typing import Any, List, Dict, Optional
import numpy as np
import random


class EXTRAWorker(BaseWorker):
    def __init__(self, worker_id: int, dim: int, lr: float, sigma: float):
        super().__init__(worker_id, dim, lr, sigma)
        self.state["weights"] = np.zeros(dim)
        self.state["grad"] = np.zeros(dim)

    def compute_local_step(self, grad: np.ndarray) -> Dict[str, np.ndarray]:
        """
        The worker computes a local (noisy) gradient for the current weights.
        In the EXTRA algorithm, the worker does not update the weights itself;
        the aggregator does that taking into account past states.
        """
        noisy_grad = grad + np.random.normal(0, self.sigma, size=grad.shape)

        self.state["grad"] = noisy_grad
        return self.state


class EXTRAOptimizer(BaseDistributedOptimizer):
    def __init__(
        self,
        dim: int,
        num_workers: int,
        lr: float,
        sigma: float = 0.01,
        topology: str = "ring",
        backup_workers: int = 0,
    ):

        self.actual_workers_count = num_workers + backup_workers
        self.required_responses = num_workers
        self.mode = "sync"

        super().__init__(dim, self.actual_workers_count, lr, sigma, topology)

        W = self.topology_matrix

        # Create the second mixing matrix \tilde{W} = (I + W) / 2
        I = np.eye(self.actual_workers_count)
        self.W_tilde = (I + W) / 2.0

        # EXTRA algorithm memory (k - previous step, k+1 - current step)
        self.X_k = np.zeros((self.actual_workers_count, dim))
        self.X_k_plus_1 = np.zeros((self.actual_workers_count, dim))
        self.G_k = np.zeros((self.actual_workers_count, dim))

        self.iteration = 0
        self.global_weights = np.zeros(dim)

    def _create_worker(
        self, worker_id: int, dim: int, lr: float, sigma: float
    ) -> BaseWorker:
        return EXTRAWorker(worker_id, dim, lr, sigma)

    def step(self, obs) -> np.ndarray:
        states = []
        for worker in self.workers:
            state = worker.compute_local_step(obs.grad)
            states.append(state)

        self._aggregate(states)
        return self.global_weights.copy()

    def _aggregate(self, states: List[Dict[str, np.ndarray]]):
        """
        Decentralized update according to EXTRA rules.
        """
        # Simulate delays
        simulated_responses = []
        for i, state in enumerate(states):
            delay = random.uniform(0.1, 1.0)
            if random.random() < 0.05:
                delay += 5.0
            simulated_responses.append((delay, i, state))
        simulated_responses.sort(key=lambda x: x[0])

        # Extract current gradients (G^{k+1})
        G_k_plus_1 = np.zeros_like(self.X_k)

        if self.mode == "sync":
            # Take only workers that finished on time
            fast_indices = [
                r[1] for r in simulated_responses[: self.required_responses]
            ]
            for idx in fast_indices:
                G_k_plus_1[idx] = states[idx]["grad"]
            # For stragglers, keep their gradient equal to their previous one (didn't finish computing)
            slow_indices = set(range(self.actual_workers_count)) - set(fast_indices)
            for idx in slow_indices:
                G_k_plus_1[idx] = self.G_k[idx]

        W = self.topology_matrix

        # === EXTRA LOGIC ===
        if self.iteration == 0:
            # Initialization
            # x^1 = W x^0 - alpha * \nabla f(x^0)
            X_next = W @ self.X_k_plus_1 - self.lr * G_k_plus_1
        else:
            # Main step
            # x^{k+2} = (I + W) x^{k+1} - \tilde{W} x^k - alpha * (G^{k+1} - G^k)
            I = np.eye(self.actual_workers_count)

            term1 = (I + W) @ self.X_k_plus_1
            term2 = self.W_tilde @ self.X_k
            term3 = self.lr * (G_k_plus_1 - self.G_k)

            X_next = term1 - term2 - term3

        # Shift "memory" one step forward
        self.X_k = self.X_k_plus_1.copy()
        self.X_k_plus_1 = X_next.copy()
        self.G_k = G_k_plus_1.copy()

        # Distribute new weights to workers
        for i in range(self.actual_workers_count):
            self.workers[i].state["weights"] = self.X_k_plus_1[i].copy()

        self.global_weights = np.mean(self.X_k_plus_1, axis=0)
        self.iteration += 1


class GradientTrackingWorker(BaseWorker):
    def __init__(self, worker_id: int, dim: int, lr: float, sigma: float):
        super().__init__(worker_id, dim, lr, sigma)
        # Worker stores only its current weights and the last computed gradient
        self.state["weights"] = np.zeros(dim)
        self.state["grad"] = np.zeros(dim)

    def compute_local_step(self, grad: np.ndarray) -> Dict[str, np.ndarray]:
        noisy_grad = grad + np.random.normal(0, self.sigma, size=grad.shape)

        self.state["grad"] = noisy_grad
        return self.state


class GradientTrackingOptimizer(BaseDistributedOptimizer):
    def __init__(
        self,
        dim: int,
        num_workers: int,
        lr: float,
        sigma: float = 0.01,
        topology: str = "ring",
    ):
        """
        Gradient Tracking. Strictly synchronous mode.
        The backup_workers mechanism is disabled because dropping a node breaks the W matrix.
        """
        # We use all workers without exceptions
        self.actual_workers_count = num_workers
        self.required_responses = num_workers

        super().__init__(dim, self.actual_workers_count, lr, sigma, topology)

        self.W = self.topology_matrix

        # Algorithm state variables
        # X: weight matrix (size: num_workers x dim)
        # Y: gradient tracker matrix (size: num_workers x dim)
        # G: matrix of past local gradients (size: num_workers x dim)
        self.X = np.zeros((self.actual_workers_count, dim))
        self.Y = np.zeros((self.actual_workers_count, dim))
        self.G = np.zeros((self.actual_workers_count, dim))

        self.iteration = 0
        self.global_weights = np.zeros(dim)

    def _create_worker(
        self, worker_id: int, dim: int, lr: float, sigma: float
    ) -> BaseWorker:
        return GradientTrackingWorker(worker_id, dim, lr, sigma)

    def step(self, obs) -> np.ndarray:
        # In strictly synchronous mode, we simply collect gradients from all workers.
        # No delay simulation — assume barrier synchronization has been passed.
        states = [worker.compute_local_step(obs.grad) for worker in self.workers]

        self._aggregate(states)
        return self.global_weights.copy()

    def _aggregate(self, states: List[Dict[str, np.ndarray]]):
        """
        Perform a Gradient Tracking step (x-update and y-update).
        """
        # Extract current gradients of all workers (this is \nabla f(X^k))
        new_G = np.stack([s["grad"] for s in states])

        if self.iteration == 0:
            # === INITIALIZATION ===
            # Gradient tracker initially equals the local gradient itself: Y^0 = G^0
            self.Y = new_G.copy()
            self.G = new_G.copy()

            # Weight update step: X^1 = W X^0 - \alpha Y^0
            self.X = self.W @ self.X - self.lr * self.Y

        else:
            # === MAIN LOOP ===

            # 1. Tracker update step (y-update)
            # Y^k = W Y^{k-1} + G^k - G^{k-1}
            self.Y = self.W @ self.Y + new_G - self.G

            # Save current gradient for the next step
            self.G = new_G.copy()

            # 2. Weight update step (x-update)
            # X^{k+1} = W X^k - \alpha Y^k
            self.X = self.W @ self.X - self.lr * self.Y

        # Distribute updated weights back to worker memory
        for i in range(self.actual_workers_count):
            self.workers[i].state["weights"] = self.X[i].copy()

        # Compute consensus (average) for monitoring in the WIND environment
        self.global_weights = np.mean(self.X, axis=0)
        self.iteration += 1


class PushPullWorker(BaseWorker):
    def __init__(self, worker_id: int, dim: int, lr: float, sigma: float):
        super().__init__(worker_id, dim, lr, sigma)
        self.state["weights"] = np.zeros(dim)
        self.state["grad"] = np.zeros(dim)

    def compute_local_step(self, grad: np.ndarray) -> Dict[str, np.ndarray]:
        noisy_grad = grad + np.random.normal(0, self.sigma, size=grad.shape)

        self.state["grad"] = noisy_grad
        return self.state


class PushPullOptimizer(BaseDistributedOptimizer):
    def __init__(
        self,
        dim: int,
        num_workers: int,
        lr: float,
        sigma: float = 0.01,
        topology: str = "ring",
    ):
        """
        Push-Pull algorithm (DIGing).
        Strictly synchronous mode for directed and undirected graphs.
        The backup_workers mechanism is disabled because dropping a node breaks the matrix.
        """
        self.actual_workers_count = num_workers
        self.required_responses = num_workers

        W_topology = topology
        if topology == "directed_ring":
            W_topology = "ring"

        super().__init__(dim, self.actual_workers_count, lr, sigma, W_topology)

        # Obtain special matrices R (Pull) and C (Push)
        self.R, self.C = TopologyManager.generate_push_pull_matrices(
            self.actual_workers_count, topology
        )

        # State variables:
        # X: weight matrix (num_workers x dim)
        # Y: gradient tracker matrix (num_workers x dim)
        # G: matrix of past local gradients (num_workers x dim)
        self.X = np.zeros((self.actual_workers_count, dim))
        self.Y = np.zeros((self.actual_workers_count, dim))
        self.G = np.zeros((self.actual_workers_count, dim))

        self.iteration = 0
        self.global_weights = np.zeros(dim)

    def _create_worker(
        self, worker_id: int, dim: int, lr: float, sigma: float
    ) -> BaseWorker:
        return PushPullWorker(worker_id, dim, lr, sigma)

    def step(self, obs) -> np.ndarray:
        # Collect gradients from all workers synchronously
        states = [worker.compute_local_step(obs.grad) for worker in self.workers]
        self._aggregate(states)
        return self.global_weights.copy()

    def _aggregate(self, states: List[Dict[str, np.ndarray]]):
        """
        Perform a Push-Pull step.
        """
        # Current local gradients: \nabla f(X^k)
        new_G = np.stack([s["grad"] for s in states])

        if self.iteration == 0:
            # INITIALIZATION
            self.Y = new_G.copy()
            self.G = new_G.copy()

            # Weight update step with Pull matrix: X^1 = R X^0 - \alpha Y^0
            self.X = self.R @ self.X - self.lr * self.Y

        else:
            # MAIN LOOP

            # 1. Tracker update step (Push mechanism with matrix C)
            # Y^k = C Y^{k-1} + G^k - G^{k-1}
            # The tracker preserves gradient mass by pushing information forward along columns.
            self.Y = self.C @ self.Y + new_G - self.G

            # Save the gradient
            self.G = new_G.copy()

            # 2. Weight update step (Pull mechanism with matrix R)
            # X^{k+1} = R X^k - \alpha Y^k
            # Workers "pull" weights toward themselves for consensus, using rows.
            self.X = self.R @ self.X - self.lr * self.Y

        # Distribute updated weights to workers
        for i in range(self.actual_workers_count):
            self.workers[i].state["weights"] = self.X[i].copy()

        # Compute consensus
        self.global_weights = np.mean(self.X, axis=0)
        self.iteration += 1
