import optuna

from utils.training.classification import ClassificationTrainer


class PruningTrainer(ClassificationTrainer):
    def __init__(self, *args, trial: optuna.Trial, **kwargs):
        kwargs.setdefault("silence", True)
        super().__init__(*args, **kwargs)
        self.trial = trial

    def validate(self) -> dict[str, float]:
        res = super().validate()
        val_acc = res.get("val_acc", 0.0)
        self.trial.report(val_acc, step=self.current_epoch)
        if self.trial.should_prune():
            raise optuna.exceptions.TrialPruned()
        return res
