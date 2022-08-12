from torch.utils.data import Dataset
from torch import Tensor, stack
import pandas as pd

class ForexData(Dataset):
    def __init__(self, seq_length, transforms=None):
        df = pd.read_csv('data/eurusd_minute.csv')
        prices = df['BidClose'].values
        pt_prices = Tensor(prices)
        self.transforms = transforms if transforms else []
        self.seq_length = seq_length
        self.series = pt_prices
        self.x = stack([self._get_dynamic(idx) for idx in range(self.series.size(0) // self.seq_length)])

    def __len__(self):
        return self.x.size(0) // self.seq_length

    def __getitem__(self, idx):
        return self.x[idx]
           
    def _get_dynamic(self, idx):
        data = self.series[idx * self.seq_length : (idx + 1) * self.seq_length]
        for transform in self.transforms:
            data = transform(data)
        return data