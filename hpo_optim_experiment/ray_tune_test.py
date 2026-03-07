import time

from ray import tune


def objective(config: dict[str, float]) -> None:
    x = config["x"]
    time.sleep(0.05)
    tune.report({"score": -(x - 2) ** 2})


if __name__ == "__main__":
    tuner = tune.Tuner(
        objective,
        param_space={"x": tune.uniform(-10, 10)},
        tune_config=tune.TuneConfig(metric="score", mode="max", num_samples=8),
    )
    results = tuner.fit()
    best_result = results.get_best_result(metric="score", mode="max")
    print(f"Best value: {best_result.metrics['score']}, Best params: {best_result.config}")