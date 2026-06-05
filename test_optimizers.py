import pytest
import numpy as np

# Import topology manager
from network.topology import TopologyManager

# Import all classical algorithms
from optimizers.classical import DSGDOptimizer, FedAvgOptimizer, ADMMOptimizer

# Import all decentralized algorithms
from optimizers.decentralized import (
    EXTRAOptimizer,
    GradientTrackingOptimizer,
    PushPullOptimizer,
)

# Import special algorithms
from optimizers.specials import CHOCOOptimizer, ADGPOptimizer, ClippedGossipOptimizer

try:
    from optimizers.specials import QuantizedPushSumOptimizer

    HAS_QUANTIZED_PUSH_SUM = True
except ImportError:
    HAS_QUANTIZED_PUSH_SUM = False

# ============================================================================
# Pytest Fixtures (Mocking the WIND benchmark environment)
# ============================================================================


class DummyObservation:
    """Emulates the Observation object provided by the WIND oracle."""

    def __init__(self, dim: int):
        self.grad = np.random.normal(0, 1.0, size=dim)


@pytest.fixture
def dim():
    return 10


@pytest.fixture
def num_workers():
    return 5


@pytest.fixture
def lr():
    return 0.1


@pytest.fixture
def dummy_obs(dim):
    """Provides a fresh observation with random gradients for testing."""
    return DummyObservation(dim)


# ============================================================================
# Unit Tests for Topology Manager
# ============================================================================


def test_topology_ring():
    """Tests the generation of a doubly stochastic ring matrix."""
    n = 5
    W = TopologyManager.generate_matrix(n, "ring")
    assert W.shape == (n, n)
    # Check doubly stochastic property (rows and cols sum to 1)
    np.testing.assert_almost_equal(np.sum(W, axis=1), np.ones(n))
    np.testing.assert_almost_equal(np.sum(W, axis=0), np.ones(n))


def test_topology_push_pull():
    """Tests the generation of asymmetric matrices for Push-Pull algorithms."""
    n = 5
    R, C = TopologyManager.generate_push_pull_matrices(n, "directed_ring")
    assert R.shape == (n, n)
    assert C.shape == (n, n)
    # R must be row-stochastic, C must be column-stochastic
    np.testing.assert_almost_equal(np.sum(R, axis=1), np.ones(n))
    np.testing.assert_almost_equal(np.sum(C, axis=0), np.ones(n))


# ============================================================================
# Unit Tests for Optimizers (Parameterized for >80% Code Coverage)
# ============================================================================

# Define all algorithms and their specific kwargs to test dynamically
ALGORITHMS_TO_TEST = [
    (DSGDOptimizer, {"topology": "ring"}),
    (FedAvgOptimizer, {"topology": "ring"}),
    (ADMMOptimizer, {"topology": "ring"}),
    (EXTRAOptimizer, {"topology": "ring"}),
    (GradientTrackingOptimizer, {"topology": "ring"}),
    (PushPullOptimizer, {"topology": "directed_ring"}),
    (CHOCOOptimizer, {"compression_ratio": 0.5}),
    (ADGPOptimizer, {"topology": "ring", "mode": "async"}),
    (ClippedGossipOptimizer, {"clip_tau": 0.5}),
]

if HAS_QUANTIZED_PUSH_SUM:
    ALGORITHMS_TO_TEST.append(
        (
            QuantizedPushSumOptimizer,
            {"topology": "directed_ring", "compression_ratio": 0.5, "gamma": 0.1},
        )
    )


@pytest.mark.parametrize("opt_class, extra_kwargs", ALGORITHMS_TO_TEST)
def test_optimizer_execution(dim, num_workers, lr, dummy_obs, opt_class, extra_kwargs):
    """
    Tests initialization and a sequence of optimization steps for all algorithms.
    This ensures that workers' compute_local_step and the _aggregate methods
    are fully covered and mathematically stable.
    """
    # 1. Test Initialization
    opt = opt_class(dim=dim, num_workers=num_workers, lr=lr, **extra_kwargs)

    assert (
        opt.actual_workers_count >= num_workers
    ), f"{opt_class.__name__} failed worker count init"
    assert (
        len(opt.workers) == opt.actual_workers_count
    ), f"{opt_class.__name__} failed worker creation"
    assert opt.global_weights.shape == (
        dim,
    ), f"{opt_class.__name__} global weights shape mismatch"

    # 2. Test Step 1 (Cold Start)
    new_weights_1 = opt.step(dummy_obs)

    assert new_weights_1.shape == (dim,), f"{opt_class.__name__} output shape mismatch"
    assert not np.isnan(
        new_weights_1
    ).any(), f"NaN values detected in {opt_class.__name__} output"

    # 3. Test Step 2 (Triggering history-dependent logic like EXTRA/PushPull)
    dummy_obs.grad = np.random.normal(0, 1.0, size=dim)  # fresh stochastic gradient
    new_weights_2 = opt.step(dummy_obs)

    assert not np.isnan(
        new_weights_2
    ).any(), f"NaN values detected in {opt_class.__name__} on step 2"


def test_local_gradient_heterogeneity(dim, num_workers, lr):
    """Tests that local gradients properly add simulated noise and fixed bias."""
    opt = DSGDOptimizer(dim=dim, num_workers=num_workers, lr=lr, sigma=0.5)
    global_grad = np.ones(dim)

    local_grads = opt._get_local_gradients(global_grad)

    assert len(local_grads) == opt.actual_workers_count
    assert local_grads[0].shape == (dim,)
    # Ensure stochastic noise is being applied (local grads shouldn't exactly match global)
    assert not np.array_equal(local_grads[0], global_grad)
