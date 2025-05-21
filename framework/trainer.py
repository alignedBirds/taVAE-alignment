from abc import ABC, abstractmethod

from torch.utils.data import Dataset, DataLoader
from framework import VariationalAutoencoder
from typing import Union, Tuple


class Trainer(ABC):
    @abstractmethod
    def train(self, dataset: Union[Dataset, DataLoader], **kwargs) -> Tuple[VariationalAutoencoder, ...]:
        pass