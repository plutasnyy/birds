from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchaudio
from matplotlib import pyplot as plt
from pytorch_lightning import LightningDataModule
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder
from torch.utils.data import DataLoader, Dataset

import torch.nn.functional as F
from torchvision.utils import make_grid


class BirdClef2022Dataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        data_path: Path,
        transformation,
        target_sample_rate: int,
        duration: int,
    ):
        self.df = df
        self.data_path = data_path
        self.transformation = transformation
        self.target_sample_rate = target_sample_rate
        self.num_samples = target_sample_rate * duration

        self.audio_paths = self.df["filename"].values
        self.label_transformer = MultiLabelBinarizer().fit([range(152)])
        numpy_labels = self.df['primary_label_encoded'].to_numpy()[:, np.newaxis]
        self.labels = self.label_transformer.transform(numpy_labels)
        # Idea: use secondary class too
        # Idea: soft labels
        # Idea: binary cross entropy
        # Idea: weight by samples by quality
        # Idea: augmentation

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, index):
        audio_path = self.data_path / "train_audio" / self.audio_paths[index]
        signal, sr = torchaudio.load(audio_path)  # loaded the audio

        # Now we first checked if the sample rate is same as TARGET_SAMPLE_RATE and if
        # it not equals we perform resampling
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
            signal = resampler(signal)

        # Next we check the number of channels of the signal
        # signal -> (num_channels, num_samples) - Eg.-(2, 14000) -> (1, 14000)
        if signal.shape[0] > 1:
            signal = torch.mean(signal, axis=0, keepdim=True)

        # Lastly we check the number of samples of the signal
        # signal -> (num_channels, num_samples) - Eg.-(1, 14000) -> (1, self.num_samples)
        # If it is more than the required number of samples, we truncate the signal
        if signal.shape[1] > self.num_samples:
            signal = signal[:, : self.num_samples]

        # If it is less than the required number of samples, we pad the signal
        if signal.shape[1] < self.num_samples:
            num_missing_samples = self.num_samples - signal.shape[1]
            last_dim_padding = (0, num_missing_samples)
            signal = F.pad(signal, last_dim_padding)

        # Finally, all the process has been done, and now we will extract mel
        # spectrogram from the signal
        mel = self.transformation(signal)
        # For pretrained models, we need 3 channel image, so for that we concatenate
        # the extracted mel
        image = torch.cat([mel, mel, mel])

        # Normalized the image
        max_val = torch.abs(image).max()
        image = image / max_val

        label = torch.FloatTensor(self.labels[index])

        return image, label


class BirdClef2022DataModule(LightningDataModule):
    def __init__(
        self,
        train_df: pd.DataFrame,
        valid_df: pd.DataFrame,
        data_path: Path,
        period: int,
        num_workers: int = 0,
        batch_size: int = 8,
    ):
        super().__init__()

        self._num_workers = num_workers
        self._batch_size = batch_size
        self.data_path = data_path
        self.period = period
        self.train_df = train_df
        self.valid_df = valid_df
        self.transformation = torchaudio.transforms.MelSpectrogram(
            sample_rate=32000, n_fft=1024, hop_length=512, n_mels=128
        )  # Idea: Try more

    def create_dataset(self, train=True):
        df = self.train_df if train else self.valid_df
        return BirdClef2022Dataset(
            df=df,
            data_path=self.data_path,
            duration=self.period,
            transformation=self.transformation,
            target_sample_rate=32000,
        )

    def train_dataloader(self):
        return self.__dataloader(train=True)

    def val_dataloader(self):
        return self.__dataloader(train=False)

    def __dataloader(self, train: bool):
        dataset = self.create_dataset(train)
        return DataLoader(
            dataset=dataset,
            batch_size=self._batch_size,
            num_workers=self._num_workers,
            shuffle=train,
            drop_last=train,
            worker_init_fn=lambda x: np.random.seed(np.random.get_state()[1][0] + x),
        )


if __name__ == "__main__":
    fold = 0
    data_path = Path("../birdclef-2022")
    df = pd.read_csv(data_path / "train_metadata_new.csv")
    train_df = df[df["fold"] != fold].reset_index(drop=True)
    valid_df = df[df["fold"] == fold].reset_index(drop=True)
    datamodule = BirdClef2022DataModule(
        data_path=data_path, train_df=train_df, valid_df=valid_df, period=7
    )

    train_dataloader = datamodule.train_dataloader()
    for batch in train_dataloader:
        imgs, labels = batch
        fig, ax = plt.subplots(1, 1, figsize=(20, 7))
        fig.suptitle("Mel Spectrogram", fontsize=15)
        ax.imshow(imgs[0].log2()[0, :, :].detach().numpy(), aspect="auto", cmap="cool")
        ax.set_title("Audio 1")
        plt.show()
        break
