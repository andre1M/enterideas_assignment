from torch.utils.data import Dataset
from torch import Tensor
from PIL import Image
import pandas as pd
import torch

from typing import Optional, Union, Callable
import os


class CustomDataset(Dataset):
    def __init__(
            self,
            image_dir: str,
            labels: pd.DataFrame,
            transform: Optional[Callable] = None
    ) -> None:
        self.labels = labels
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: Union[Tensor, list, int]) -> dict:
        if torch.is_tensor(idx):
            idx = idx.tolist()

        label = 'georges' if self.labels.iloc[idx, 1] == 1 else 'non_georges'
        image_name = os.path.join(
            self.image_dir, label, self.labels.iloc[idx, 0]
        )
        image = Image.open(image_name).convert('RGB')
        label = torch.Tensor([self.labels.iloc[idx, 1]]).float()

        if self.transform:
            image = self.transform(image)

        return {'image': image, 'label': label}
