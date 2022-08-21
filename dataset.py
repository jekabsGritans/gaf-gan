from torch.utils.data import Dataset
import numpy as np
from transforms import rel_gaf, simple_raster
import torch

class ForexData(Dataset):
    def __init__(self, prices, seq_length, encoding='gaf'):
        tensor = self.get_interval_tensor(prices, seq_length)
        if encoding == 'gaf':
            tensor = rel_gaf(tensor)
        elif encoding == 'simple':
            tensor = simple_raster(tensor)
        else:
            raise ValueError('Invalid encoding')

        # Remove matrices where nan is present
        tensor = tensor[~torch.isnan(tensor.view(tensor.size(0),-1)).any(axis=1)]
        self.x = tensor 

    def get_interval_tensor(self, prices, seq_length):
        prices = np.array(prices)
        log_prices = np.log(prices)
        log_diffs = np.diff(log_prices)
        log_diffs = log_diffs[:len(log_diffs) // seq_length * seq_length]
        tensor = torch.from_numpy(log_diffs).float().view(-1, seq_length)

        # Stretch to [-1,1]
        tensor /= tensor.abs().max(dim=1).values.view(-1, 1)
        return tensor

    def __len__(self):
        return self.x.size(0) 

    def __getitem__(self, idx):
        out = self.x[idx]
        return out