from pathlib import Path

import pandas as pd
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from src.datamodule import BirdClef2022DataModule
from src.lit_module import BirdClef2022Model

seed_everything(0)
fold = 0
data_path = Path("../birdclef-2022")
df = pd.read_csv(data_path / "train_metadata_new.csv")
train_df = df[df["fold"] != fold].reset_index(drop=True)
valid_df = df[df["fold"] == fold].reset_index(drop=True)
datamodule = BirdClef2022DataModule(
    data_path=data_path, train_df=train_df, valid_df=valid_df, period=7, num_workers=1
)
model = BirdClef2022Model()
lr_monitor = LearningRateMonitor()
loss_checkpoint = ModelCheckpoint(
    filename="best_loss",
    monitor="valid_loss",
    save_top_k=1,
    mode="min",
)

f1_checkpoint = ModelCheckpoint(
    filename="best_f1",
    monitor="valid_f1_score",
    save_top_k=1,
    mode="max",
)

trainer = Trainer(
    # precision=16,
    gpus=0,
    max_epochs=5,
    callbacks=[
        loss_checkpoint,
        f1_checkpoint,
        lr_monitor,
    ],
    fast_dev_run=25,
)

trainer.fit(model, datamodule=datamodule)
