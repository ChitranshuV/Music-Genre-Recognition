# !pip3 install torch torchvision torchaudio librosa
# !apt-get install libsndfile1-dev
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import pandas as pd
import torchaudio


# Genre_top and Genre_Class relation
# ['Electronic', 'Experimental', 'Folk', 'Hip-Hop', 'Instrumental', 'International', 'Pop', 'Rock']
# [0 1 2 3 4 5 6 7]


class FMADataset(Dataset):
    def __init__(
        self,
        annotations_file,
        audio_dir,
        target_sample_rate,
        num_samples,
        input_image_size,
        device,
        phase,
    ):
        df = pd.read_csv(annotations_file)

        self.phase = phase

        if self.phase == "Train":
            self.annotations = df[df["train-test"] == "Train"]
        if self.phase == "Test":
            self.annotations = df[df["train-test"] == "Test"]

        self.audio_dir = audio_dir
        self.device = device
        self.target_sample_rate = target_sample_rate
        self.num_samples = num_samples
        self.input_image_size = input_image_size

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        audio_sample_path = self._get_audio_sample_path(index)
        label = self._get_audio_sample_label(index)
        signal, sr = torchaudio.load(audio_sample_path)

        signal = self._resample_if_necessary(signal, sr)
        signal = self._mix_down_if_necessary(signal)
        signal = self._cut_if_necessary(signal)
        signal = self._right_pad_if_necessary(signal)

        signal = signal.to(self.device)
        signal = self._spectrogram(signal)
        spectrogram_image = self._spectrogram_transform(signal)

        return spectrogram_image, label

    def _spectrogram(self, signal):
        mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.target_sample_rate,
            n_fft=1024,
            hop_length=512,
            n_mels=self.input_image_size,
        ).to(self.device)
        return mel_spectrogram(signal)

    def _spectrogram_transform(self, signal):
        image = torch.stack((signal[0],) * 3, axis=-1).permute(2, 0, 1)
        print(image.shape)
        # Image is of (3,299,1292) shape now
        image = transforms.ToPILImage()(image)
        # image.show()
        image = transforms.RandomCrop(self.input_image_size)(image)
        image = transforms.ToTensor()(image)
        image = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(
            image
        )

        return image

    def _cut_if_necessary(self, signal):
        if signal.shape[1] > self.num_samples:
            signal = signal[:, : self.num_samples]
        return signal

    def _right_pad_if_necessary(self, signal):
        length_signal = signal.shape[1]
        if length_signal < self.num_samples:
            num_missing_samples = self.num_samples - length_signal
            last_dim_padding = (0, num_missing_samples)
            signal = torch.nn.functional.pad(signal, last_dim_padding)
        return signal

    def _resample_if_necessary(self, signal, sr):
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
            signal = resampler(signal)
        return signal

    def _mix_down_if_necessary(self, signal):
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal

    def _get_audio_sample_path(self, index):
        path = os.path.join(self.audio_dir, self.annotations.iloc[index, 0])
        return path

    def _get_audio_sample_label(self, index):
        audio_label = torch.tensor(float(self.annotations.iloc[index, 6]))
        return audio_label


if __name__ == "__main__":
    ANNOTATIONS_FILE = "/run/media/high/Edu/Music Genre Classification Project/Music-Genre-Recognition/track-genre.csv"
    AUDIO_DIR = "/run/media/high/Edu/Music Genre Classification Project/fma_small/"
    SAMPLE_RATE = 22050
    NUM_SAMPLES = SAMPLE_RATE * 20
    PHASE = "Train"
    IMAGE_SIZE = 299
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using device {device}")

    fma = FMADataset(
        ANNOTATIONS_FILE,
        AUDIO_DIR,
        SAMPLE_RATE,
        NUM_SAMPLES,
        IMAGE_SIZE,
        device,
        PHASE,
    )
    print(f"There are {len(fma)} samples in the {PHASE} dataset.")

    output_tensor = fma[125][0]
    print(output_tensor.shape)

    numarray = output_tensor.permute(1, 2, 0).detach().cpu().numpy()
    print(numarray.shape)
    # to convert into numpy array of 229,229,3 size. Channel should be last for PIL

    img = Image.fromarray(numarray.astype("uint8"), mode="RGB")
    img.show()
