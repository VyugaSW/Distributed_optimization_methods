================================================================================
          Decentralized Optimization in Dynamic Environments
================================================================================

# OVERVIEW
This repository contains a comprehensive simulation framework for evaluating 
decentralized and distributed optimization algorithms in dynamic environments 
(where the global optimum is continuously shifting) via WIND benchmark (https://github.com/muffin003/WIND). 

Unlike standard static machine learning benchmarks, this project focuses on 
real-world network constraints, including data heterogeneity, communication 
bottlenecks, asymmetric network topologies, and asynchronous updates.

# FEATURES
- Dynamic Environment Simulation: The global optimum drifts continuously, 
  forcing algorithms into a steady-state tracking error rather than zero error.
- Comprehensive Algorithm Suite: 11 distinct distributed algorithms spanning 
  classical methods, gradient tracking, and specialized robust protocols.
- Network Topologies: Support for ring, star, directed ring, and fully 
  connected graphs via `TopologyManager`.
- Advanced Metrics: Built-in assessment of Tracking Error, Lyapunov metrics, 
  Consensus Error (agent variance), Communication Costs, and Computation load.
- Production-Ready: Includes a non-intrusive logging system and extensive 
  unit tests (>90% coverage) using `pytest`.

================================================================================
# ALGORITHMS IMPLEMENTED
================================================================================
The algorithms are divided into three logical groups:

1. Classical Methods (optimizers/classical.py)
   - DSGD (Decentralized SGD): Baseline consensus method.
   - FedAvg (Federated Averaging): Client-server architecture with E local steps.
   - ADMM (Alternating Direction Method of Multipliers): Optimization via 
     Lagrangian penalties.

2. Gradient Tracking Methods (optimizers/decentralized.py)
   - EXTRA: Exact first-order algorithm utilizing memory to eliminate variance.
   - GradTracking: Uses an auxiliary variable to track the global gradient direction.
   - PushPull: Designed for directed graphs, decoupling weight pulling and 
     gradient pushing.

3. Specialized & Robust Methods (optimizers/specials.py)
   - CHOCO-SGD: Uses Top-K compression to reduce network traffic by 90%.
   - ADGP (Asynchronous Distributed Gradient Push): Eliminates idle waiting by 
     allowing asynchronous updates.
   - Clipped Gossip: A Byzantine-robust method that clips excessive weight 
     updates to defend against malicious or broken nodes.
   - Quantized Push-Sum: Combines directed graph support with extreme data 
     compression and consensus stabilization.
   - Consensus SPSA: A zeroth-order (gradient-free) method that estimates 
     gradients using simultaneous random perturbations.

================================================================================
# PROJECT STRUCTURE
================================================================================
WIND_Benchmark/
|-- src/                    # Core benchmark components (Environment, Oracle, Metrics)
|-- network/
|   `-- topology.py         # Generates stochastic and directed matrices
|-- optimizers/
|   |-- base_optimizer.py   # Base abstract classes for Workers and Optimizers
|   |-- classical.py        # Baseline methods
|   |-- decentralized.py    # Gradient tracking methods
|   `-- specials.py         # Advanced/Robust methods
|-- logger_utils.py         # Decorator for deep optimization logging
|-- test_optimizers.py      # Pytest fixtures and unit tests
|-- main.py                 # Entry point to run the 1000-iteration benchmark
`-- README.txt              # This file

================================================================================
# INSTALLATION & REQUIREMENTS
================================================================================
Python 3.8+ is required.

Install the required dependencies:
> pip install numpy pandas tabulate pytest pytest-cov

================================================================================
# USAGE GUIDE
================================================================================

1. RUNNING THE BENCHMARK
To execute the main simulation (T=1000 iterations, dim=10) and generate the 
final Pareto-efficiency table:
> python main.py

2. VIEWING THE LOGS
The project uses a built-in logger. By default, it prints basic progress to the 
console. For deep analysis of a specific algorithm (e.g., viewing consensus 
error at every step):
- Open `logger_utils.py`
- Change `level=logging.INFO` to `level=logging.DEBUG`
- Re-run `python main.py`

3. RUNNING TESTS
To verify the mathematical integrity of all 11 algorithms and ensure that no 
network matrices are broken, run the test suite with coverage:
> pytest test_optimizers.py -v --cov=optimizers

================================================================================
# UNDERSTANDING THE METRICS TABLE
================================================================================
When you run `main.py`, a table is generated. Here is how to interpret it:

- error_l2: The average distance from the network's center of mass to the moving 
  optimum. Due to dynamic equilibrium, all accurate consensus methods will 
  converge to a similar theoretical limit.
- worst_agent_deviation: (Consensus Error). The distance of the most disjointed 
  worker from the center of mass. Lower is better (EXTRA/GradTracking excel here).
- grad_computations: Computational cost. SPSA uses 2x, FedAvg uses E times more.
- transmitted_floats: Communication bandwidth used. CHOCO and Quantized Push-Sum 
  will show significantly lower numbers here due to compression.
