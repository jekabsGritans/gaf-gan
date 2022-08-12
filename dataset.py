from torch.utils.data import Dataset
from torch import Tensor, stack, isnan
import pandas as pd
import numpy as np

from transforms import pt_gaf, stretch

def unsqueeze(x):
    return x.unsqueeze(0)

class ForexData(Dataset):
    def __init__(self, seq_length, transforms=[stretch, pt_gaf, unsqueeze]):
        df = pd.read_csv('data/eurusd_minute.csv')
        prices = df['BidClose'].values
        # log_prices = np.log(prices)
        pt_prices= Tensor(prices)
        self.transforms = transforms if transforms else []
        self.seq_length = seq_length
        self.series = pt_prices

        stackable = [self._get_dynamic(idx) for idx in range(self.series.size(0) // self.seq_length)]
        stackable = list(filter(lambda x: x is not None, stackable))
        self.x = stack(stackable)

    def __len__(self):
        return self.x.size(0) // self.seq_length

    def __getitem__(self, idx):
        return self.x[idx]
           
    def _get_dynamic(self, idx):
        data = self.series[idx * self.seq_length : (idx + 1) * self.seq_length]
        for transform in self.transforms:
            data = transform(data)
        return data if not isnan(data).any() else None