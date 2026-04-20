# -*- coding: cp1251 -*-
from optimizers.classical import DSGDOptimizer, ADMMOptimizer
import numpy as np
import random
import numpy as np
from src import (
    make_environment, 
    make_oracle, 
    MetricsCollection, 
    TrackingErrorMetric, 
    BenchmarkRunner,
    GaussianNoise,
    DynamicRegretMetric
)

def setup_wind_environment(dim=2, T=100, drift_type="linear", noise_sigma=0.01, seed=42):
    """
    Настраивает среду (Environment) и Оракула для бенчмарка.
    """
    # 1. Конфигурация среды
    env_config = {
        "dim": dim,
        "drift": {"type": drift_type, "velocity": [0.01] * dim},
        "landscape": {"type": "quadratic", "condition_number": 1.0},
        "noise": {"type": "gaussian", "sigma": noise_sigma}
    }
    
    env = make_environment(env_config, seed=seed)
    
    # 2. Настройка оракула (First-Order для градиентных методов)
    # Используем значение шума из конфига среды
    value_noise = GaussianNoise(sigma=noise_sigma, seed=seed)
    oracle = make_oracle("first-order", env, value_noise=value_noise, seed=seed)
    
    # 3. Набор метрик для анализа
    metrics = MetricsCollection([
        TrackingErrorMetric(norm="l2"),
        DynamicRegretMetric()
    ])
    
    return env, oracle, metrics

def run_custom_algorithm(optimizer, env, oracle, metrics, T=100, x0=None, seed=42):
    if x0 is None:
        x0 = np.zeros(env.dim)
        
    runner = BenchmarkRunner(
        environment=env,
        oracle=oracle,
        metrics=metrics,
        record_trajectory=True
    )
    
    result = runner.run(optimizer, T=T, x0=x0, seed=seed)
    
    print(f"Эксперимент завершен: {optimizer.__class__.__name__}")
    
    # Печатаем всё содержимое final_metrics
    if result.final_metrics:
        print("Итоговые показатели:")
        for metric_name, value in result.final_metrics.items():
            print(f"{metric_name}: {value:.6f}")
    else:
        print("Словарь final_metrics пуст. Проверь настройки MetricsCollection.")
    
    return result

if __name__ == "__main__":
    # Пример запуска
    dim = 100
    T = 200
    seed = 123

    # 1. Создаем компоненты WIND
    env, oracle, metrics = setup_wind_environment(dim=dim, T=T, seed=seed)

    # 2. Инициализируем твой алгоритм (например, Push-Pull)
    # Убедись, что параметры dim и lr соответствуют задаче
    my_optimizer = DSGDOptimizer(dim=dim, num_workers=4, lr=0.4, topology="all-to-all")
    my_optimizer_admm = ADMMOptimizer(dim=dim, num_workers=400, lr=0.4, topology="star")

    # 3. Запускаем
    result = run_custom_algorithm(my_optimizer_admm, env, oracle, metrics, T=T, seed=seed)