import optuna
import time

def objective(trial: optuna.Trial) -> float:
    x = trial.suggest_float('x', -10, 10)
    trial.suggest_loguniform
    time.sleep(0.05)  # Simulate a costly computation
    return (x - 2) ** 2


if __name__ == "__main__":
    study = optuna.create_study()
    study.optimize(objective, n_trials=100, show_progress_bar=True)
    print(f"Best value: {study.best_value}, Best params: {study.best_params}")