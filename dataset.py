from torch.utils.data import Dataset
import numpy as np
from transforms import pt_gaf, pt_noisy_gaf, simple_raster, stretch
import torch

class ForexData(Dataset):
    def __init__(self, prices, seq_length, encoding='gaf'):
        prices = np.array(prices)
        values = np.log(prices)
        values = values[:len(values) // seq_length * seq_length]
        tensor = torch.from_numpy(values).float().view(-1, seq_length)
        tensor = stretch(tensor)
        if encoding == 'gaf':
            tensor = pt_gaf(tensor)
        elif encoding == 'noisy_gaf':
            tensor = pt_noisy_gaf(tensor)
        elif encoding == 'simple':
            tensor = simple_raster(tensor)
        else:
            raise ValueError('Invalid encoding')

        # Remove matrices where nan is present
        tensor = tensor[~torch.isnan(tensor.view(tensor.size(0),-1)).any(axis=1)]
        self.x = tensor.view(-1, 1, seq_length, seq_length)

    def __len__(self):
        return self.x.size(0) 

    def __getitem__(self, idx):
        out = self.x[idx]
        return out


