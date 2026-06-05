# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from tabulate import tabulate

from src import (
    make_environment,
    make_oracle,
    MetricsCollection,
    TrackingErrorMetric,
    LyapunovMetric,
    NormalizedLyapunovMetric,
    InstantaneousLossMetric,
    DynamicRegretMetric,
    BenchmarkRunner,
    GaussianNoise,
)

from optimizers.classical import DSGDOptimizer, ADMMOptimizer, FedAvgOptimizer
from optimizers.decentralized import (
    EXTRAOptimizer,
    GradientTrackingOptimizer,
    PushPullOptimizer,
)
from optimizers.specials import (
    CHOCOOptimizer,
    ADGPOptimizer,
    ClippedGossipOptimizer,
    QuantizedPushSumOptimizer,
    ConsensusSPSAOptimizer,  # <-- Added SPSA import
)


def get_fresh_components(dim, T, noise_sigma, seed):
    """
    Creates new, clean instances of the environment, oracle, and metrics.
    Using the same seed ensures that each algorithm faces an identical
    target trajectory (drift).
    """
    env_config = {
        "dim": dim,
        "drift": {"type": "linear", "velocity": [0.05] * dim},
        "landscape": {"type": "quadratic", "condition_number": 1.0},
        "noise": {"type": "gaussian", "sigma": noise_sigma},
    }

    env = make_environment(env_config, seed=seed)

    value_noise = GaussianNoise(sigma=noise_sigma, seed=seed)
    oracle = make_oracle("first-order", env, value_noise=value_noise, seed=seed)

    rho_val = 1.0
    metrics = MetricsCollection(
        [
            TrackingErrorMetric(norm="l2"),
            InstantaneousLossMetric(),
            DynamicRegretMetric(),
            LyapunovMetric(rho=rho_val),
            NormalizedLyapunovMetric(rho=rho_val),
        ]
    )

    return env, oracle, metrics


def run_experiment(name, optimizer, dim, T, noise_sigma, seed):
    env, oracle, metrics = get_fresh_components(dim, T, noise_sigma, seed)

    x0 = np.zeros(dim)

    runner = BenchmarkRunner(
        environment=env, oracle=oracle, metrics=metrics, record_trajectory=True
    )

    result = runner.run(optimizer, T=T, x0=x0, seed=seed)

    final_stats = result.final_metrics
    final_stats["Algorithm"] = name

    global_w = optimizer.global_weights

    consensus_error = np.mean(
        [
            np.linalg.norm(worker.state["weights"] - global_w)
            for worker in optimizer.workers
        ]
    )

    final_stats["consensus_error"] = consensus_error
    # =================================================================

    return final_stats


if __name__ == "__main__":
    dim = 5
    T = 1000
    noise_sigma = 0.01
    seed = 123

    configs = [
        ("DSGD", DSGDOptimizer, {"num_workers": 50, "lr": 0.1, "topology": "ring"}),
        ("FedAvg", FedAvgOptimizer, {"num_workers": 50, "lr": 0.1, "topology": "ring"}),
        ("ADMM", ADMMOptimizer, {"num_workers": 50, "lr": 0.1, "topology": "ring"}),
        ("EXTRA", EXTRAOptimizer, {"num_workers": 50, "lr": 0.1, "topology": "ring"}),
        (
            "GradTracking",
            GradientTrackingOptimizer,
            {"num_workers": 50, "lr": 0.1, "topology": "ring"},
        ),
        (
            "PushPull",
            PushPullOptimizer,
            {"num_workers": 50, "lr": 0.1, "topology": "directed_ring"},
        ),
        (
            "CHOCO-SGD",
            CHOCOOptimizer,
            {"num_workers": 50, "lr": 0.1, "gamma": 0.1, "compression_ratio": 0.1},
        ),
        (
            "ADGP",
            ADGPOptimizer,
            {"num_workers": 50, "lr": 0.1, "topology": "ring", "mode": "async"},
        ),
        (
            "ClippedGossip",
            ClippedGossipOptimizer,
            {"num_workers": 50, "lr": 0.1, "topology": "ring", "clip_tau": 0.5},
        ),
        (
            "QuantizedPushSum",
            QuantizedPushSumOptimizer,
            {
                "num_workers": 50,
                "lr": 0.1,
                "topology": "directed_ring",
                "compression_ratio": 0.1,
                "gamma": 0.1,
            },
        ),
        # <-- Added the new SPSA method
        (
            "DSPSA",
            ConsensusSPSAOptimizer,
            {"num_workers": 50, "lr": 0.1, "topology": "ring"},
        ),
    ]

    all_results = []
    print(f"=== Running WIND Benchmark (T={T}, dim={dim}) ===")

    for name, opt_class, params in configs:
        print(f"Testing: {name:15s}...", end=" ", flush=True)
        try:
            opt_instance = opt_class(dim=dim, **params)
            stats = run_experiment(name, opt_instance, dim, T, noise_sigma, seed)

            global_w = opt_instance.global_weights
            consensus_error = np.max(
                [
                    np.linalg.norm(worker.state["weights"] - global_w)
                    for worker in opt_instance.workers
                ]
            )
            stats["worst_agent_deviation"] = consensus_error

            if hasattr(opt_instance, "E"):
                stats["grad_computations"] = T * opt_instance.E
            else:
                # SPSA is zeroth-order, it performs 2 function evaluations per step
                if "SPSA" in name:
                    stats["grad_computations"] = T * 2
                else:
                    stats["grad_computations"] = T

            comm_cost = T * dim

            if "CHOCO" in name:
                ratio = getattr(
                    opt_instance, "ratio", params.get("compression_ratio", 1.0)
                )
                comm_cost = T * dim * ratio
            elif "PushPull" in name:
                comm_cost = T * dim * 2

            stats["transmitted_floats"] = comm_cost

            all_results.append(stats)
            print("Done.")
        except Exception as e:
            print(f"ERROR: {e}")

    if all_results:
        df = pd.DataFrame(all_results)

        columns_to_show = [
            "Algorithm",
            "error_l2",
            "worst_agent_deviation",
            "grad_computations",
            "transmitted_floats",
        ]

        df_filtered = df[columns_to_show]
        print("\n=== ALGORITHM PERFORMANCE ANALYSIS ===")
        print(
            tabulate(
                df_filtered,
                headers="keys",
                tablefmt="grid",
                showindex=False,
                floatfmt=".4f",
            )
        )
