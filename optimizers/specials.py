# -*- coding: cp1251 -*-
from optimizers.base_optimizer import BaseWorker, BaseDistributedOptimizer
from network.topology import TopologyManager
from utils.logger_utils import log_optimization_step
from typing import Any, List, Dict
import numpy as np
import random


class ADGPWorker(BaseWorker):
    """
    Worker node for the Asynchronous Distributed Gradient Push (ADGP) algorithm.
    Stores the local gradient for further aggregation; the main logic (including buffers)
    is handled by the optimizer to facilitate memory and delay simulation.
    """

    def __init__(self, worker_id: int, dim: int, lr: float, sigma: float):
        super().__init__(worker_id, dim, lr, sigma)
        self.state["weights"] = np.zeros(dim)
        self.state["grad"] = np.zeros(dim)

    def compute_local_step(
        self, local_grad: np.ndarray, **kwargs
    ) -> Dict[str, np.ndarray]:
        self.state["grad"] = local_grad
        return self.state


class ADGPOptimizer(BaseDistributedOptimizer):
    """
    Asynchronous Distributed Gradient Push (ADGP) Optimizer.
    Implements the Push-Sum protocol with asynchronous gradient tracking.
    """

    def __init__(
        self,
        dim: int,
        num_workers: int,
        lr: float,
        sigma: float = 0.01,
        topology: str = "ring",
        mode: str = "async",
        backup_workers: int = 0,
    ):
        self.actual_workers_count = num_workers + backup_workers
        self.required_responses = num_workers
        self.mode = mode

        W_topology = topology
        if topology == "ring":
            W_topology = "directed_ring"

        super().__init__(dim, num_workers, lr, sigma, W_topology)

        _, self.A = TopologyManager.generate_push_pull_matrices(
            self.actual_workers_count, W_topology
        )

        self.X = np.zeros((self.actual_workers_count, dim))
        self.Y = np.ones((self.actual_workers_count, 1))
        self.Z = np.zeros((self.actual_workers_count, dim))
        self.W_trk = np.zeros((self.actual_workers_count, dim))

        self.V = np.zeros((self.actual_workers_count, dim))
        self.U = np.ones((self.actual_workers_count, 1))

        self.last_grad = np.zeros((self.actual_workers_count, dim))

        self.Rho = np.zeros((self.actual_workers_count, self.actual_workers_count, dim))
        self.Rho_tilde = np.zeros(
            (self.actual_workers_count, self.actual_workers_count, dim)
        )

        self.iteration = 0
        self.global_weights = np.zeros(dim)

    def _create_worker(
        self, worker_id: int, dim: int, lr: float, sigma: float
    ) -> BaseWorker:
        return ADGPWorker(worker_id, dim, lr, sigma)

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
        new_grads = np.stack([s["grad"] for s in states])

        simulated_responses = []
        for i in range(self.actual_workers_count):
            delay = random.uniform(0.1, 1.0)
            if random.random() < 0.05:
                delay += 5.0
            simulated_responses.append((delay, i, new_grads[i]))

        simulated_responses.sort(key=lambda x: x[0])

        if self.mode == "sync":
            if self.iteration == 0:
                self.W_trk = new_grads.copy()
                self.last_grad = new_grads.copy()
            else:
                diff_grad = new_grads - self.last_grad
                W_half = self.W_trk + diff_grad
                self.W_trk = self.A @ W_half
                self.last_grad = new_grads.copy()

            self.V = self.X - self.lr * self.W_trk
            self.U = self.Y.copy()

            self.X = self.A @ self.V
            self.Y = self.A @ self.U
            self.Z = self.X / self.Y

        elif self.mode == "async":
            for _, idx, grad in simulated_responses:
                if self.iteration == 0:
                    self.W_trk[idx] = grad.copy()
                    self.last_grad[idx] = grad.copy()
                else:
                    diff_grad_i = grad - self.last_grad[idx]

                    rho_diff = np.zeros(self.dim)
                    for j in range(self.actual_workers_count):
                        if j != idx and self.A[idx, j] > 0:
                            rho_diff += self.Rho[idx, j] - self.Rho_tilde[idx, j]

                    w_half_i = self.W_trk[idx] + rho_diff + diff_grad_i

                    for j in range(self.actual_workers_count):
                        if j != idx and self.A[j, idx] > 0:
                            self.Rho[j, idx] += self.A[j, idx] * w_half_i

                    self.W_trk[idx] = self.A[idx, idx] * w_half_i

                    for j in range(self.actual_workers_count):
                        if j != idx and self.A[idx, j] > 0:
                            self.Rho_tilde[idx, j] = self.Rho[idx, j].copy()

                    self.last_grad[idx] = grad.copy()

                v_i = self.X[idx] - self.lr * self.W_trk[idx]
                u_i = self.Y[idx].copy()

                self.V[idx] = v_i
                self.U[idx] = u_i

                x_i = self.A[idx, idx] * v_i
                y_i = self.A[idx, idx] * u_i

                for j in range(self.actual_workers_count):
                    if j != idx and self.A[idx, j] > 0:
                        x_i += self.A[idx, j] * self.V[j]
                        y_i += self.A[idx, j] * self.U[j]

                self.X[idx] = x_i
                self.Y[idx] = y_i
                self.Z[idx] = x_i / y_i

        for i in range(self.actual_workers_count):
            self.workers[i].state["weights"] = self.Z[i].copy()

        self.global_weights = np.mean(self.Z, axis=0)
        self.iteration += 1


class CHOCOWorker(BaseWorker):
    """
    Worker node for the CHOCO-SGD algorithm.
    Responsible for computing the local stochastic gradient step.
    """

    def __init__(self, worker_id: int, dim: int, lr: float, sigma: float):
        super().__init__(worker_id, dim, lr, sigma)
        self.state["weights"] = np.zeros(dim)

    def compute_local_step(
        self, local_grad: np.ndarray, **kwargs
    ) -> Dict[str, np.ndarray]:
        self.state["weights"] -= self.lr * local_grad
        return self.state


class CHOCOOptimizer(BaseDistributedOptimizer):
    """
    CHOCO-SGD Optimizer.
    Implements decentralized optimization with arbitrary communication compression
    and error-compensation memory (X_hat).
    """

    def __init__(
        self,
        dim: int,
        num_workers: int,
        lr: float,
        sigma: float = 0.01,
        topology: str = "ring",
        gamma: float = 0.1,
        compression_ratio: float = 0.01,
    ):
        self.actual_workers_count = num_workers
        super().__init__(dim, num_workers, lr, sigma, topology)

        self.gamma = gamma
        self.compression_ratio = compression_ratio

        self.W = self.topology_matrix

        self.X_hat = np.zeros((self.actual_workers_count, dim))

        self.iteration = 0
        self.global_weights = np.zeros(dim)

    def _create_worker(
        self, worker_id: int, dim: int, lr: float, sigma: float
    ) -> BaseWorker:
        return CHOCOWorker(worker_id, dim, lr, sigma)

    def _compress_top_k(self, delta: np.ndarray, ratio: float) -> np.ndarray:
        """
        Applies a Top-K sparsification operator.
        Only sends the coordinates with the largest absolute values.
        """
        k = max(1, int(delta.shape[1] * ratio))
        compressed = np.zeros_like(delta)

        for i in range(delta.shape[0]):
            idx = np.argsort(np.abs(delta[i]))[-k:]
            compressed[i, idx] = delta[i, idx]

        return compressed

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
        X_half = np.stack([s["weights"] for s in states])

        consensus_update = self.gamma * (self.W @ self.X_hat - self.X_hat)
        X_new = X_half + consensus_update

        delta = X_new - self.X_hat
        Q_matrix = self._compress_top_k(delta, self.compression_ratio)

        self.X_hat += Q_matrix

        for i in range(self.actual_workers_count):
            self.workers[i].state["weights"] = X_new[i].copy()

        self.global_weights = np.mean(X_new, axis=0)
        self.iteration += 1


class ClippedGossipWorker(BaseWorker):
    def compute_local_step(
        self, local_grad: np.ndarray, **kwargs
    ) -> Dict[str, np.ndarray]:
        self.state["last_grad"] = local_grad
        return self.state


class ClippedGossipOptimizer(BaseDistributedOptimizer):
    """
    Byzantine-Robust Decentralized SGD based on Clipped Gossip.
    """

    def __init__(
        self,
        dim: int,
        num_workers: int,
        lr: float,
        clip_tau: float = 0.5,
        sigma: float = 0.01,
        topology: str = "ring",
        mode: str = "sync",
    ):
        self.clip_tau = clip_tau
        self.actual_workers_count = num_workers
        self.mode = mode
        super().__init__(dim, num_workers, lr, sigma, topology)
        self.global_weights = np.zeros(dim)

    def _create_worker(
        self, worker_id: int, dim: int, lr: float, sigma: float
    ) -> BaseWorker:
        return ClippedGossipWorker(worker_id, dim, lr, sigma)

    @log_optimization_step
    def step(self, obs) -> np.ndarray:
        local_grads = self._get_local_gradients(obs.grad)
        states = [
            w.compute_local_step(local_grads[i]) for i, w in enumerate(self.workers)
        ]
        self._aggregate(states)
        return self.global_weights.copy()

    def _aggregate(self, states: List[Dict[str, np.ndarray]]):
        X = np.stack([w.state["weights"] for w in self.workers])
        G = np.stack([s["last_grad"] for s in states])
        W = self.topology_matrix

        X_next = np.zeros_like(X)
        for i in range(self.actual_workers_count):
            consensus_sum = np.zeros(self.dim)
            for j in range(self.actual_workers_count):
                if W[i, j] > 0 and i != j:
                    diff = X[j] - X[i]
                    norm = np.linalg.norm(diff)
                    if norm > self.clip_tau:
                        diff = diff * (self.clip_tau / norm)
                    consensus_sum += W[i, j] * diff

            X_next[i] = X[i] + consensus_sum - self.lr * G[i]

        for i in range(self.actual_workers_count):
            self.workers[i].state["weights"] = X_next[i].copy()

        self.global_weights = np.mean(X_next, axis=0)


class QuantizedPushSumWorker(BaseWorker):
    def __init__(self, worker_id: int, dim: int, lr: float, sigma: float):
        super().__init__(worker_id, dim, lr, sigma)
        # Инициализация переменных Push-Sum
        self.state["y"] = 1.0  # Вес узла
        self.state["w"] = np.zeros(dim)  # Вектор состояния

    def compute_local_step(self, local_grad: np.ndarray, **kwargs) -> Dict[str, np.ndarray]:
        self.state["last_grad"] = local_grad
        return self.state

class QuantizedPushSumOptimizer(BaseDistributedOptimizer):
    """
    Quantized Push-Sum for Directed Graphs.
    """
    def __init__(self, dim: int, num_workers: int, lr: float, compression_ratio: float = 0.1, gamma: float = 0.1, sigma: float = 0.01, topology: str = "directed_ring"):
        self.actual_workers_count = num_workers
        self.ratio = compression_ratio
        self.gamma = gamma
        super().__init__(dim, num_workers, lr, sigma, topology)
        self.C = self.topology_matrix 
        
        self.global_weights = np.zeros(dim)
        self.Q = np.zeros((num_workers, dim)) # Память квантования
        self.iteration = 0

    def _create_worker(self, worker_id: int, dim: int, lr: float, sigma: float) -> BaseWorker:
        return QuantizedPushSumWorker(worker_id, dim, lr, sigma)

    def _compress_top_k(self, delta: np.ndarray, ratio: float) -> np.ndarray:
        k = max(1, int(delta.shape[1] * ratio))
        compressed = np.zeros_like(delta)
        for i in range(delta.shape[0]):
            idx = np.argsort(np.abs(delta[i]))[-k:]
            compressed[i, idx] = delta[i, idx]
        return compressed

    @log_optimization_step
    def step(self, obs) -> np.ndarray:
        local_grads = self._get_local_gradients(obs.grad)
        states = [w.compute_local_step(local_grads[i]) for i, w in enumerate(self.workers)]
        self._aggregate(states)
        return self.global_weights.copy()

    def _aggregate(self, states: List[Dict[str, np.ndarray]]):
        W_current = np.stack([w.state["w"] for w in self.workers])
        Y_current = np.array([w.state["y"] for w in self.workers])
        G = np.stack([s["last_grad"] for s in states])

        # Квантование (Сжатие разницы)
        delta = W_current - self.Q
        compressed_delta = self._compress_top_k(delta, self.ratio)
        self.Q += compressed_delta

        # Стабилизированный Push-Sum консенсус с шагом gamma
        # Вместо резкого прыжка, мы аккуратно примешиваем квантованные сообщения
        W_next = W_current + self.gamma * (self.C @ self.Q - self.Q) - self.lr * G
        
        # Переменная Y также должна обновляться с учетом гаммы для баланса
        Y_next = Y_current + self.gamma * (self.C @ Y_current - Y_current)

        # Восстановление реальных весов x = w / y
        X_next = W_next / Y_next[:, None]

        for i in range(self.actual_workers_count):
            self.workers[i].state["w"] = W_next[i].copy()
            self.workers[i].state["y"] = Y_next[i]
            self.workers[i].state["weights"] = X_next[i].copy()

        self.global_weights = np.mean(X_next, axis=0)
        self.iteration += 1



class ConsensusSPSAWorker(BaseWorker):
    def compute_local_step(self, local_grad: np.ndarray, **kwargs) -> Dict[str, np.ndarray]:
        delta = np.random.choice([-1.0, 1.0], size=self.dim)
        
        spsa_grad = np.dot(local_grad, delta) * delta
        
        self.state["last_grad"] = spsa_grad
        return self.state

class ConsensusSPSAOptimizer(BaseDistributedOptimizer):
    def __init__(self, dim: int, num_workers: int, lr: float, sigma: float = 0.01, topology: str = "ring"):
        super().__init__(dim, num_workers, lr, sigma, topology)
        self.global_weights = np.zeros(dim)

    def _create_worker(self, worker_id: int, dim: int, lr: float, sigma: float) -> BaseWorker:
        return ConsensusSPSAWorker(worker_id, dim, lr, sigma)

    def step(self, obs) -> np.ndarray:
        local_grads = self._get_local_gradients(obs.grad)
        states = [w.compute_local_step(local_grads[i]) for i, w in enumerate(self.workers)]
        self._aggregate(states)
        return self.global_weights.copy()

    def _aggregate(self, states: List[Dict[str, np.ndarray]]):
        X = np.stack([w.state["weights"] for w in self.workers])
        G = np.stack([s["last_grad"] for s in states])
        W = self.topology_matrix

        X_next = W @ X - self.lr * G

        for i in range(self.actual_workers_count):
            self.workers[i].state["weights"] = X_next[i].copy()
            
        self.global_weights = np.mean(X_next, axis=0)