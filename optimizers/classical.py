# -*- coding: cp1251 -*-
from optimizers.base_optimizer import BaseWorker, BaseDistributedOptimizer
from network.topology import TopologyManager
from typing import Any, List, Dict, Optional
import numpy as np
import random


# Distributed SGD
class DSGDWorker(BaseWorker):
    def compute_local_step(self, grad: np.ndarray) -> Dict[str, np.ndarray]:
        """
        The worker computes the gradient (local step).
        In the Parameter Server architecture, it does not update the weights itself,
        but passes the gradient to the server for aggregation.
        """
        # grad is the exact gradient
        # We add noise to simulate computation on a "subset" of data
        stochastic_noise = np.random.normal(0, self.sigma, size=grad.shape)
        self.state["last_grad"] = grad + stochastic_noise
        return self.state


class DSGDOptimizer(BaseDistributedOptimizer):
    def __init__(
        self,
        dim: int,
        num_workers: int,
        lr: float,
        sigma: float = 0.01,
        topology: str = "star",
        mode: str = "sync",
        backup_workers: int = 0,
    ):
        """
        :param mode: "sync" (synchronous) or "async" (asynchronous)
        :param backup_workers: number of additional agents to handle stragglers
        """
        # Actual number of workers
        self.actual_workers_count = num_workers + backup_workers

        # The server needs to wait for exactly 'num_workers' responses per step
        self.required_responses = num_workers
        self.mode = mode

        super().__init__(dim, self.actual_workers_count, lr, sigma, topology)

        # Global weights stored on the server
        self.server_weights = np.zeros(dim)

    def _create_worker(
        self, worker_id: int, dim: int, lr: float, sigma: float
    ) -> BaseWorker:
        return DSGDWorker(worker_id, dim, lr, sigma)

    def _build_topology(self, topology: str) -> np.ndarray:
        """
        Override the method: generate matrix W
        taking into account the total number of workers (including backups).
        """
        return TopologyManager.generate_matrix(self.actual_workers_count, mode=topology)

    def step(self, obs: Any) -> np.ndarray:
        """
        1. Local Computation: each worker performs calculations based on its own data.
        2. Network Communication: simulate data exchange via _aggregate.
        3. State Update: workers receive updated data.
        """

        new_states = []
        for worker in self.workers:
            state = worker.compute_local_step(obs.grad)
            new_states.append(state)

        mixed_states = self._aggregate(new_states)

        all_weights = []
        for i, worker in enumerate(self.workers):
            worker.update_state(mixed_states[i])
            all_weights.append(worker.state["weights"])

        return np.mean(all_weights, axis=0)

    def _aggregate(
        self, states: List[Dict[str, np.ndarray]]
    ) -> List[Dict[str, np.ndarray]]:
        """
        Aggregation using the topology matrix W, accounting for delays.
        """
        # 1. Simulate delays (Latency)
        simulated_responses = []
        for i, state in enumerate(states):
            delay = random.uniform(0.1, 1.0)
            if random.random() < 0.1:
                delay += 5.0
            simulated_responses.append((delay, i, state))

        # Sort by arrival time
        simulated_responses.sort(key=lambda x: x[0])

        # Prepare weight matrices (X) and gradients (G)
        X = np.stack(
            [self.workers[i].state["weights"] for i in range(self.actual_workers_count)]
        )
        G = np.zeros_like(X)

        W = self.topology_matrix

        if self.mode == "sync":
            # 2. Backup agent mechanism
            # Take only the first N workers
            accepted_indices = [
                r[1] for r in simulated_responses[: self.required_responses]
            ]

            for idx in accepted_indices:
                G[idx] = states[idx]["last_grad"]

            # 3. Local step (compute intermediate weights Y)
            # Nodes that did not make it have zero gradient in G and keep their weights
            Y = X - self.lr * G

            # 4. Topology mixing: X_next = W @ Y
            # Each node i receives a weighted sum of neighbors' weights j according to W[i,j]
            X_next = W @ Y

            # Update global consensus for WIND reporting
            self.server_weights = np.mean(X_next, axis=0)

            return [{"weights": X_next[i]} for i in range(self.actual_workers_count)]

        elif self.mode == "async":
            # In asynchronous mode, each arriving gradient immediately affects the node
            # and is partially propagated to neighbors via the row of matrix W
            X_async = X.copy()

            for _, idx, state in simulated_responses:
                # Local update of the specific worker
                grad = state["last_grad"]
                X_async[idx] -= self.lr * grad

                # Immediate mixing of the updated node with neighbors (row of matrix)
                # This simulates instant data transmission in a decentralized network
                X_async[idx] = W[idx] @ X_async

            self.server_weights = np.mean(X_async, axis=0)
            return [{"weights": X_async[i]} for i in range(self.actual_workers_count)]


# ADMM
class ADMMWorker(BaseWorker):
    def __init__(self, worker_id: int, dim: int, lr: float, sigma: float):
        super().__init__(worker_id, dim, lr, sigma)
        # Each worker now has its own set of variables
        self.state["x"] = np.zeros(dim)
        self.state["u"] = np.zeros(dim)
        self.state["z_local"] = np.zeros(
            dim
        )  # Its own version of the consensus variable (for decentralized)

    def compute_local_step(self, grad: np.ndarray, rho: float) -> Dict[str, np.ndarray]:
        """
        Step 1: x-update.
        Uses local z_local.
        """
        local_grad = grad + np.random.normal(0, self.sigma, size=grad.shape)

        # Compute the gradient of the augmented Lagrangian
        grad_aug_lagrangian = local_grad + rho * (
            self.state["x"] - self.state["z_local"] + self.state["u"]
        )

        # Update local weights
        self.state["x"] -= self.lr * grad_aug_lagrangian

        return self.state

    def update_dual(self):
        """
        Step 3: u-update.
        Updates relative to local z_local after exchange with neighbors.
        """
        self.state["u"] += self.state["x"] - self.state["z_local"]


class ADMMOptimizer(BaseDistributedOptimizer):
    def __init__(
        self,
        dim: int,
        num_workers: int,
        lr: float,
        rho: float = 1.0,
        sigma: float = 0.01,
        topology: str = "ring",
        mode: str = "sync",
        backup_workers: int = 0,
    ):
        self.rho = rho
        self.actual_workers_count = num_workers + backup_workers
        self.required_responses = num_workers
        self.mode = mode

        # Global z is needed ONLY for reporting to the WIND environment
        # (the algorithm itself no longer needs it for computations)
        self.z_global = np.zeros(dim)

        super().__init__(dim, self.actual_workers_count, lr, sigma, topology)

    def _build_topology(self, topology: str) -> np.ndarray:
        return TopologyManager.generate_matrix(self.actual_workers_count, mode=topology)

    def _create_worker(
        self, worker_id: int, dim: int, lr: float, sigma: float
    ) -> BaseWorker:
        return ADMMWorker(worker_id, dim, lr, sigma)

    def step(self, obs) -> np.ndarray:
        # Step 1: Local computations (x-update)
        states = []
        for worker in self.workers:
            # No longer passing z_global!
            state = worker.compute_local_step(obs.grad, self.rho)
            states.append(state)

        # Step 2: Decentralized exchange (z-update via matrix W)
        self._aggregate(states)

        # Step 3: Dual variable update (u-update)
        for worker in self.workers:
            worker.update_dual()

        return self.z_global.copy()

    def _aggregate(self, states: List[Dict[str, np.ndarray]]):
        """
        Gossip mixing over the network graph (matrix W).
        """
        # Simulate delays
        simulated_responses = []
        for i, state in enumerate(states):
            delay = random.expovariate(1.0)
            if random.random() < 0.05:
                delay += 5.0
            simulated_responses.append((delay, i, state))

        simulated_responses.sort(key=lambda x: x[0])

        # Extract all vectors x_i into matrix X
        X = np.stack([s["x"] for s in states])
        W = self.topology_matrix

        if self.mode == "sync":
            fast_indices = [
                r[1] for r in simulated_responses[: self.required_responses]
            ]

            # Ideally, slow agents should use their old weights.
            # For simplicity of modeling, we take X as is, but the essence of decentralization is multiplication:
            # Each node receives a weighted sum ONLY from its neighbors.
            Z_next = W @ X

            # Distribute new personal z_local back to workers
            for i in range(self.actual_workers_count):
                self.workers[i].state["z_local"] = Z_next[i].copy()

            # Compute the cluster average point just to return it from the step method
            self.z_global = np.mean(Z_next, axis=0)

        elif self.mode == "async":
            # Asynchronous Gossip: nodes exchange data as they become ready
            Z_async = np.stack([w.state["z_local"] for w in self.workers])

            for _, idx, state in simulated_responses:
                # Update row X only for the worker that arrived
                X[idx] = state["x"]
                Z_async[idx] = W[idx] @ X

            for i in range(self.actual_workers_count):
                self.workers[i].state["z_local"] = Z_async[i].copy()

            self.z_global = np.mean(Z_async, axis=0)


# Federated Average
class FedAvgWorker(BaseWorker):
    def __init__(self, worker_id: int, dim: int, lr: float, sigma: float):
        super().__init__(worker_id, dim, lr, sigma)
        # Each client has its own local weights
        self.state["weights"] = np.zeros(dim)

    def compute_local_epochs(
        self, base_grad: np.ndarray, E: int
    ) -> Dict[str, np.ndarray]:
        """
        Instead of a single step (as in DSGD), the client performs E local updates.
        """
        local_w = self.state["weights"].copy()

        for _ in range(E):
            # Simulate gradient computation on a new mini-batch
            # (in reality this would be an honest pass through the client's data)
            noisy_grad = base_grad + np.random.normal(
                0, self.sigma, size=base_grad.shape
            )

            # Local descent step
            local_w -= self.lr * noisy_grad

        # Save the result after E epochs
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

        super().__init__(dim, self.actual_workers_count, lr, sigma, topology)

    def _build_topology(self, topology: str) -> np.ndarray:
        return TopologyManager.generate_matrix(self.actual_workers_count, mode=topology)

    def _create_worker(
        self, worker_id: int, dim: int, lr: float, sigma: float
    ) -> BaseWorker:
        return FedAvgWorker(worker_id, dim, lr, sigma)

    def step(self, obs) -> np.ndarray:
        # 1. Local Update phase: clients train autonomously for E epochs
        states = []
        for worker in self.workers:
            state = worker.compute_local_epochs(obs.grad, self.E)
            states.append(state)

        # 2. Communication phase: mixing over the network topology
        self._aggregate(states)

        return self.global_weights.copy()

    def _aggregate(self, states: List[Dict[str, np.ndarray]]):
        """
        Decentralized model averaging taking into account the matrix W.
        """
        # Simulate delays (some took longer to compute their E epochs)
        simulated_responses = []
        for i, state in enumerate(states):
            delay = random.uniform(
                1.0, 5.0
            )  # Delays in FedAvg are typically larger because more computation is involved
            if random.random() < 0.1:
                delay += 10.0
            simulated_responses.append((delay, i, state))

        simulated_responses.sort(key=lambda x: x[0])

        # Collect local weights into matrix X
        X = np.stack([s["weights"] for s in states])
        W = self.topology_matrix

        if self.mode == "sync":
            fast_indices = [
                r[1] for r in simulated_responses[: self.required_responses]
            ]

            # In classic FedAvg, client weights are averaged proportionally to
            # their dataset sizes (n_k / n). Our matrix W automatically
            # serves as these weight coefficients!

            # Mixing (Gossip / Federation)
            X_next = W @ X

            # Distribute the averaged weights back to clients
            for i in range(self.actual_workers_count):
                self.workers[i].state["weights"] = X_next[i].copy()

            self.global_weights = np.mean(X_next, axis=0)

        elif self.mode == "async":
            # Asynchronous FedAvg (e.g., FedBuff)
            X_async = X.copy()
            for _, idx, state in simulated_responses:
                X_async[idx] = state["weights"]
                X_async[idx] = W[idx] @ X_async

            for i in range(self.actual_workers_count):
                self.workers[i].state["weights"] = X_async[i].copy()

            self.global_weights = np.mean(X_async, axis=0)
