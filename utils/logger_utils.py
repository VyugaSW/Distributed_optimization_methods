import logging
import numpy as np
from functools import wraps

# ============================================================================
# Logging Configuration
# ============================================================================
# Set how often the logger should print output (e.g., 100 = every 100th iteration)
LOG_FREQUENCY = 100
LOG_LEVEL = logging.INFO  # /logging.DEBUG

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def log_optimization_step(func):
    """
    Decorator to wrap the step() method of optimizers.
    It logs the iteration number, global weight norm, and consensus error
    without modifying the underlying mathematical logic of the algorithms.
    """

    @wraps(func)
    def wrapper(self, obs, *args, **kwargs):
        # 1. Initialize logger and iteration counter if they don't exist
        if not hasattr(self, "logger"):
            self.logger = logging.getLogger(self.__class__.__name__)
        if not hasattr(self, "iteration_count"):
            self.iteration_count = 0
            self.logger.debug(f"Initialized logging for {self.__class__.__name__}")

        self.iteration_count += 1

        # 2. Execute the original mathematical step
        result = func(self, obs, *args, **kwargs)

        # 3. Check if we should log this iteration (always log the 1st step and every Nth step)
        if self.iteration_count == 1 or self.iteration_count % LOG_FREQUENCY == 0:

            # Calculate metrics ONLY when logging to save computational overhead
            w_norm = np.linalg.norm(self.global_weights)

            # Calculate consensus error (max deviation from mean) if workers exist
            consensus_err = 0.0
            if (
                hasattr(self, "workers")
                and len(self.workers) > 0
                and "weights" in self.workers[0].state
            ):
                try:
                    X = np.stack([w.state["weights"] for w in self.workers])
                    mean_x = np.mean(X, axis=0)
                    consensus_err = float(np.max(np.linalg.norm(X - mean_x, axis=1)))
                except Exception:
                    consensus_err = float("nan")

            # 4. Log the output
            # INFO: General progress
            self.logger.info(
                f"Iteration: {self.iteration_count:4d} | Global Weights Norm: {w_norm:.4f}"
            )

            # DEBUG: Highly detailed metrics for deep analysis
            self.logger.debug(
                f"Detailed Stats - Iteration: {self.iteration_count} | "
                f"Consensus Error (Variance): {consensus_err:.6f} | "
                f"Input Global Grad Norm: {np.linalg.norm(obs.grad):.4f}"
            )

        return result

    return wrapper
