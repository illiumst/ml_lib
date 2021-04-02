import torch
from pytorch_lightning import Callback, Trainer, LightningModule


class BestScoresCallback(Callback):

    def __init__(self, *monitors) -> None:
        super().__init__()
        self.monitors = list(*monitors)

        self.best_scores = {monitor: 0.0 for monitor in self.monitors}
        self.best_epoch = {monitor: 0 for monitor in self.monitors}

    def on_validation_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        epoch = pl_module.current_epoch

        for monitor in self.best_scores.keys():
            current_score = trainer.callback_metrics.get(monitor)
            if current_score is None:
                pass
            elif torch.isinf(current_score):
                pass
            elif torch.isnan(current_score):
                pass
            else:
                self.best_scores[monitor] = max(self.best_scores[monitor], current_score)
                if self.best_scores[monitor] == current_score:
                    self.best_epoch[monitor] = max(self.best_epoch[monitor], epoch)
