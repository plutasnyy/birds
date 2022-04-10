from functools import partial

import numpy as np
from pytorch_lightning import LightningModule
from scipy.optimize import minimize
from sklearn.metrics import f1_score
from torch import nn
from torch.optim import Adam
from torchvision import models


class BirdCLEFResnet(nn.Module):
    def __init__(self):
        super(BirdCLEFResnet, self).__init__()
        self.base_model = models.__getattribute__("resnet50")(pretrained=True)
        for param in self.base_model.parameters():
            param.requires_grad = False

        in_features = self.base_model.fc.in_features

        self.base_model.fc = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, 152),
        )

    def forward(self, x):
        return self.base_model(x)


class ThresholdOptimizer:
    def __init__(self, loss_fn):
        self.coef_ = {}
        self.loss_fn = loss_fn
        self.coef_["x"] = [0.5]

    def _loss(self, coef, X, y):
        ll = self.loss_fn(y, X, coef)
        return -ll

    def fit(self, X, y):
        loss_partial = partial(self._loss, X=X, y=y)
        initial_coef = [0.5]
        self.coef_ = minimize(loss_partial, initial_coef, method="nelder-mead")

    def coefficients(self):
        return self.coef_["x"]

    def calc_score(self, X, y, coef):
        return self.loss_fn(y, X, coef)


def row_wise_f1_score_micro(y_true, y_pred, threshold=0.5):
    def event_thresholder(x, threshold):
        return x > threshold

    return f1_score(
        y_true=y_true, y_pred=event_thresholder(y_pred, threshold), average="macro"
    )


class BirdClef2022Model(LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.thresholder = ThresholdOptimizer(row_wise_f1_score_micro)

        self.model = BirdCLEFResnet()
        self.loss = nn.BCEWithLogitsLoss()
        self.save_hyperparameters()

    def forward(self, image):
        """Forward pass. Returns logits."""
        return self.model(image)

    def _proces_batch(self, batch, batch_idx):
        data, labels = batch
        outputs = self.model(data)
        loss = self.loss(outputs, labels)
        return dict(
            loss=loss,
            preds=outputs.tolist(),
            labels=labels.tolist(),
        )

    def training_step(self, batch, batch_idx):
        return self._proces_batch(batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        return self._proces_batch(batch, batch_idx)

    def _summarize_results(self, outputs, stage):
        y_pred, y_true = [], []
        for output in outputs:
            y_true.extend(output['labels'])
            y_pred.extend(output['preds'])

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        self.thresholder.fit(y_pred, y_true)
        coef = self.thresholder.coefficients().item()
        f1_score = self.thresholder.calc_score(y_pred, y_true, coef)
        f1_score_05 = self.thresholder.calc_score(y_pred, y_true, [0.5])
        f1_score_03 = self.thresholder.calc_score(y_pred, y_true, [0.3])

        log_data = {
            f"{stage}_coef": coef,
            f"{stage}_f1_score": f1_score,
            f"{stage}_f1_score_05": f1_score_05,
            f"{stage}_f1_score_03": f1_score_03,
        }
        self.log_dict(log_data, prog_bar=True)

    def training_epoch_end(self, training_step_outputs):
        return self._summarize_results(training_step_outputs, stage="train")

    def validation_epoch_end(self, validation_step_outputs):
        return self._summarize_results(validation_step_outputs, stage="valid")

    def configure_optimizers(self):
        optimizer = Adam(self.model.parameters(), lr=1e-3)
        # Idea: some loss scheduler
        return optimizer
