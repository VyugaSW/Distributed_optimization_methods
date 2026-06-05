# -*- coding: cp1251 -*-
from optimizers.base_optimizer import BaseWorker, BaseDistributedOptimizer
from network.topology import TopologyManager
from utils.logger_utils import log_optimization_step
from typing import Any, List, Dict
import numpy as np
import random


# ==========================================
# EXTRA
# ==========================================
class EXTRAWorker(BaseWorker):
    def __init__(self, worker_id: int, dim: int, lr: float, sigma: float):
        super().__init__(worker_id, dim, lr, sigma)
        self.state["weights"] = np.zeros(dim)
        self.state["grad"] = np.zeros(dim)

    def compute_local_step(
        self, local_grad: np.ndarray, **kwargs
    ) -> Dict[str, np.ndarray]:
        """
        Worker computes its true local gradient (which already contains heterogeneity).
        No additional stochastic noise is needed if we simulate exact full gradients.
        """
        self.state["grad"] = local_grad
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

        super().__init__(dim, num_workers, lr, sigma, topology)

        W = self.topology_matrix
        I = np.eye(self.actual_workers_count)
        self.W_tilde = (I + W) / 2.0

        self.X_k = np.zeros((self.actual_workers_count, dim))
        self.X_k_plus_1 = np.zeros((self.actual_workers_count, dim))
        self.G_k = np.zeros((self.actual_workers_count, dim))

        self.iteration = 0
        self.global_weights = np.zeros(dim)

    def _create_worker(
        self, worker_id: int, dim: int, lr: float, sigma: float
    ) -> BaseWorker:
        return EXTRAWorker(worker_id, dim, lr, sigma)

    @log_optimization_step
    def step(self, obs) -> np.ndarray:
        local_grads = self._get_local_gradients(obs.grad)

        states = []
        for i, worker in enumerate(self.workers):
            state = worker.compute_local_step(local_grads[i])
            states.append(state)

        self._aggregate(states)
        return self.global_weights.copy()

    def _aggregate(self, states: List[Dict[str, np.ndarray]]):
        simulated_responses = []
        for i, state in enumerate(states):
            delay = random.uniform(0.1, 1.0)
            if random.random() < 0.05:
                delay += 5.0
            simulated_responses.append((delay, i, state))
        simulated_responses.sort(key=lambda x: x[0])

        G_k_plus_1 = np.zeros_like(self.X_k)

        if self.mode == "sync":
            fast_indices = [
                r[1] for r in simulated_responses[: self.required_responses]
            ]
            for idx in fast_indices:
                G_k_plus_1[idx] = states[idx]["grad"]
            slow_indices = set(range(self.actual_workers_count)) - set(fast_indices)
            for idx in slow_indices:
                G_k_plus_1[idx] = self.G_k[idx]

        W = self.topology_matrix

        if self.iteration == 0:
            X_next = W @ self.X_k_plus_1 - self.lr * G_k_plus_1
        else:
            I = np.eye(self.actual_workers_count)
            term1 = (I + W) @ self.X_k_plus_1
            term2 = self.W_tilde @ self.X_k
            term3 = self.lr * (G_k_plus_1 - self.G_k)
            X_next = term1 - term2 - term3

        self.X_k = self.X_k_plus_1.copy()
        self.X_k_plus_1 = X_next.copy()
        self.G_k = G_k_plus_1.copy()

        for i in range(self.actual_workers_count):
            self.workers[i].state["weights"] = self.X_k_plus_1[i].copy()

        self.global_weights = np.mean(self.X_k_plus_1, axis=0)
        self.iteration += 1


# ==========================================
# Gradient Tracking (D-GET)
# ==========================================
class GradientTrackingWorker(BaseWorker):
    def __init__(self, worker_id: int, dim: int, lr: float, sigma: float):
        super().__init__(worker_id, dim, lr, sigma)
        self.state["weights"] = np.zeros(dim)
        self.state["grad"] = np.zeros(dim)

    def compute_local_step(
        self, local_grad: np.ndarray, **kwargs
    ) -> Dict[str, np.ndarray]:
        self.state["grad"] = local_grad
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
        self.actual_workers_count = num_workers
        self.required_responses = num_workers
        super().__init__(dim, num_workers, lr, sigma, topology)

        self.W = self.topology_matrix
        self.X = np.zeros((self.actual_workers_count, dim))
        self.Y = np.zeros((self.actual_workers_count, dim))
        self.G = np.zeros((self.actual_workers_count, dim))

        self.iteration = 0
        self.global_weights = np.zeros(dim)

    def _create_worker(
        self, worker_id: int, dim: int, lr: float, sigma: float
    ) -> BaseWorker:
        return GradientTrackingWorker(worker_id, dim, lr, sigma)

    @log_optimization_step
    def step(self, obs) -> np.ndarray:
        local_grads = self._get_local_gradients(obs.grad)

        states = [
            self.workers[i].compute_local_step(local_grads[i])
            for i in range(self.actual_workers_count)
        ]
        self._aggregate(states)
        return self.global_weights.copy()

    def _aggregate(self, states: List[Dict[str, np.ndarray]]):
        new_G = np.stack([s["grad"] for s in states])

        if self.iteration == 0:
            self.Y = new_G.copy()
            self.G = new_G.copy()
            self.X = self.W @ self.X - self.lr * self.Y
        else:
            self.Y = self.W @ self.Y + new_G - self.G
            self.G = new_G.copy()
            self.X = self.W @ self.X - self.lr * self.Y

        for i in range(self.actual_workers_count):
            self.workers[i].state["weights"] = self.X[i].copy()

        self.global_weights = np.mean(self.X, axis=0)
        self.iteration += 1


# ==========================================
# Push-Pull (DIGing)
# ==========================================
class PushPullWorker(BaseWorker):
    def __init__(self, worker_id: int, dim: int, lr: float, sigma: float):
        super().__init__(worker_id, dim, lr, sigma)
        self.state["weights"] = np.zeros(dim)
        self.state["grad"] = np.zeros(dim)

    def compute_local_step(
        self, local_grad: np.ndarray, **kwargs
    ) -> Dict[str, np.ndarray]:
        self.state["grad"] = local_grad
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
        self.actual_workers_count = num_workers
        self.required_responses = num_workers

        W_topology = topology
        if topology == "directed_ring":
            W_topology = "ring"

        super().__init__(dim, num_workers, lr, sigma, W_topology)

        self.R, self.C = TopologyManager.generate_push_pull_matrices(
            self.actual_workers_count, topology
        )

        self.X = np.zeros((self.actual_workers_count, dim))
        self.Y = np.zeros((self.actual_workers_count, dim))
        self.G = np.zeros((self.actual_workers_count, dim))

        self.iteration = 0
        self.global_weights = np.zeros(dim)

    def _create_worker(
        self, worker_id: int, dim: int, lr: float, sigma: float
    ) -> BaseWorker:
        return PushPullWorker(worker_id, dim, lr, sigma)

    @log_optimization_step
    def step(self, obs) -> np.ndarray:
        local_grads = self._get_local_gradients(obs.grad)

        states = [
            self.workers[i].compute_local_step(local_grads[i])
            for i in range(self.actual_workers_count)
        ]
        self._aggregate(states)
        return self.global_weights.copy()

    def _aggregate(self, states: List[Dict[str, np.ndarray]]):
        new_G = np.stack([s["grad"] for s in states])

        if self.iteration == 0:
            self.Y = new_G.copy()
            self.G = new_G.copy()
            self.X = self.R @ self.X - self.lr * self.Y
        else:
            self.Y = self.C @ self.Y + new_G - self.G
            self.G = new_G.copy()
            self.X = self.R @ self.X - self.lr * self.Y

        for i in range(self.actual_workers_count):
            self.workers[i].state["weights"] = self.X[i].copy()

        self.global_weights = np.mean(self.X, axis=0)
        self.iteration += 1
