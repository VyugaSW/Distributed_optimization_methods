# -*- coding: cp1251 -*-
from optimizers.base_optimizer import BaseWorker, BaseDistributedOptimizer
from network.topology import TopologyManager
from typing import Any, List, Dict
from utils.logger_utils import log_optimization_step
import numpy as np
import random


# ==========================================
# Distributed SGD
# ==========================================
class DSGDWorker(BaseWorker):
    def compute_local_step(
        self, local_grad: np.ndarray, **kwargs
    ) -> Dict[str, np.ndarray]:
        """
        The worker saves its localized gradient.
        """
        self.state["last_grad"] = local_grad
        return self.state


class DSGDOptimizer(BaseDistributedOptimizer):
    def __init__(self, dim: int, num_workers: int, lr: float, sigma: float = 0.01, topology: str = "star", mode: str = "sync", backup_workers: int = 0):
        self.actual_workers_count = num_workers + backup_workers
        self.required_responses = num_workers
        self.mode = mode
        super().__init__(dim, num_workers, lr, sigma, topology)
        self.global_weights = np.zeros(dim)

    def _create_worker(self, worker_id: int, dim: int, lr: float, sigma: float) -> BaseWorker:
        return DSGDWorker(worker_id, dim, lr, sigma)

    @log_optimization_step
    def step(self, obs: Any) -> np.ndarray:
        local_grads = self._get_local_gradients(obs.grad)
        new_states = []
        for i, worker in enumerate(self.workers):
            state = worker.compute_local_step(local_grads[i])
            new_states.append(state)

        mixed_states = self._aggregate(new_states)
        all_weights = []
        for i, worker in enumerate(self.workers):
            worker.update_state(mixed_states[i])
            all_weights.append(worker.state["weights"])

        self.global_weights = np.mean(all_weights, axis=0)
        return self.global_weights.copy()

    def _aggregate(self, states: List[Dict[str, np.ndarray]]) -> List[Dict[str, np.ndarray]]:
        simulated_responses = []
        for i, state in enumerate(states):
            delay = random.uniform(0.1, 1.0)
            if random.random() < 0.1:
                delay += 5.0
            simulated_responses.append((delay, i, state))

        simulated_responses.sort(key=lambda x: x[0])
        X = np.stack([self.workers[i].state["weights"] for i in range(self.actual_workers_count)])
        G = np.zeros_like(X)
        W = self.topology_matrix

        if self.mode == "sync":
            accepted_indices = [r[1] for r in simulated_responses[: self.required_responses]]
            for idx in accepted_indices:
                G[idx] = states[idx]["last_grad"]
            Y = X - self.lr * G
            X_next = W @ Y
            return [{"weights": X_next[i]} for i in range(self.actual_workers_count)]

        elif self.mode == "async":
            X_async = X.copy()
            for _, idx, state in simulated_responses:
                grad = state["last_grad"]
                X_async[idx] -= self.lr * grad
                X_async[idx] = W[idx] @ X_async
            return [{"weights": X_async[i]} for i in range(self.actual_workers_count)]



# ==========================================
# ADMM
# ==========================================
class ADMMWorker(BaseWorker):
    def __init__(self, worker_id: int, dim: int, lr: float, sigma: float):
        super().__init__(worker_id, dim, lr, sigma)
        self.state["x"] = np.zeros(dim)
        self.state["u"] = np.zeros(dim)
        self.state["z_local"] = np.zeros(dim)

    def compute_local_step(
        self, local_grad: np.ndarray, rho: float = 1.0, **kwargs
    ) -> Dict[str, np.ndarray]:
        # Compute the gradient of the augmented Lagrangian using the localized gradient
        grad_aug_lagrangian = local_grad + rho * (
            self.state["x"] - self.state["z_local"] + self.state["u"]
        )
        self.state["x"] -= self.lr * grad_aug_lagrangian
        # ИСПРАВЛЕНО: Синхронизируем weights, чтобы бенчмарк мог корректно считывать локальное состояние
        self.state["weights"] = self.state["x"].copy()
        return self.state

    def update_dual(self):
        self.state["u"] += self.state["x"] - self.state["z_local"]


class ADMMOptimizer(BaseDistributedOptimizer):
    def __init__(self, dim: int, num_workers: int, lr: float, rho: float = 1.0, sigma: float = 0.01, topology: str = "ring", mode: str = "sync", backup_workers: int = 0):
        self.rho = rho
        self.actual_workers_count = num_workers + backup_workers
        self.required_responses = num_workers
        self.mode = mode
        self.global_weights = np.zeros(dim)
        super().__init__(dim, num_workers, lr, sigma, topology)

    def _create_worker(self, worker_id: int, dim: int, lr: float, sigma: float) -> BaseWorker:
        return ADMMWorker(worker_id, dim, lr, sigma)
    
    @log_optimization_step
    def step(self, obs) -> np.ndarray:
        local_grads = self._get_local_gradients(obs.grad)
        states = []
        for i, worker in enumerate(self.workers):
            state = worker.compute_local_step(local_grads[i], rho=self.rho)
            states.append(state)

        self._aggregate(states)
        for worker in self.workers:
            worker.update_dual()

        return self.global_weights.copy()

    def _aggregate(self, states: List[Dict[str, np.ndarray]]):
        simulated_responses = []
        for i, state in enumerate(states):
            delay = random.expovariate(1.0)
            if random.random() < 0.05:
                delay += 5.0
            simulated_responses.append((delay, i, state))
        simulated_responses.sort(key=lambda x: x[0])

        X = np.stack([s["x"] for s in states])
        W = self.topology_matrix

        if self.mode == "sync":
            Z_next = W @ X
            for i in range(self.actual_workers_count):
                self.workers[i].state["z_local"] = Z_next[i].copy()
            self.global_weights = np.mean(Z_next, axis=0)

        elif self.mode == "async":
            Z_async = np.stack([w.state["z_local"] for w in self.workers])
            for _, idx, state in simulated_responses:
                X[idx] = state["x"]
                Z_async[idx] = W[idx] @ X
            for i in range(self.actual_workers_count):
                self.workers[i].state["z_local"] = Z_async[i].copy()
            self.global_weights = np.mean(Z_async, axis=0)

# ==========================================
# Federated Averaging
# ==========================================
class FedAvgWorker(BaseWorker):
    def __init__(self, worker_id: int, dim: int, lr: float, sigma: float):
        super().__init__(worker_id, dim, lr, sigma)
        self.state["weights"] = np.zeros(dim)

    def compute_local_step(
        self, local_grad_base: np.ndarray, E: int = 5, **kwargs
    ) -> Dict[str, np.ndarray]:
        local_w = self.state["weights"].copy()
        for _ in range(E):
            # We add stochastic noise at each epoch over the base local gradient
            noisy_grad = local_grad_base + np.random.normal(
                0, self.sigma, size=local_grad_base.shape
            )
            local_w -= self.lr * noisy_grad

        self.state["weights"] = local_w
        return self.state


class FedAvgOptimizer(BaseDistributedOptimizer):
    def __init__(
        self,
        dim: int,
        num_workers: int,
        lr: float,
        E: int = 5,
        sigma: float = 0.01,
        topology: str = "star",
        mode: str = "sync",
        backup_workers: int = 0,
    ):
        self.E = E
        self.actual_workers_count = num_workers + backup_workers
        self.required_responses = num_workers
        self.mode = mode
        self.global_weights = np.zeros(dim)

        super().__init__(dim, num_workers, lr, sigma, topology)

    def _create_worker(
        self, worker_id: int, dim: int, lr: float, sigma: float
    ) -> BaseWorker:
        return FedAvgWorker(worker_id, dim, lr, sigma)

    @log_optimization_step
    def step(self, obs) -> np.ndarray:
        local_grads = self._get_local_gradients(obs.grad)

        states = []
        for i, worker in enumerate(self.workers):
            state = worker.compute_local_step(local_grads[i], E=self.E)
            states.append(state)

        self._aggregate(states)
        return self.global_weights.copy()

    def _aggregate(self, states: List[Dict[str, np.ndarray]]):
        simulated_responses = []
        for i, state in enumerate(states):
            delay = random.uniform(1.0, 5.0)
            if random.random() < 0.1:
                delay += 10.0
            simulated_responses.append((delay, i, state))
        simulated_responses.sort(key=lambda x: x[0])

        X = np.stack([s["weights"] for s in states])
        W = self.topology_matrix

        if self.mode == "sync":
            X_next = W @ X
            for i in range(self.actual_workers_count):
                self.workers[i].state["weights"] = X_next[i].copy()
            self.global_weights = np.mean(X_next, axis=0)

        elif self.mode == "async":
            X_async = X.copy()
            for _, idx, state in simulated_responses:
                X_async[idx] = state["weights"]
                X_async[idx] = W[idx] @ X_async
            for i in range(self.actual_workers_count):
                self.workers[i].state["weights"] = X_async[i].copy()
            self.global_weights = np.mean(X_async, axis=0)