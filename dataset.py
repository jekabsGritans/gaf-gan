from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from transforms import pt_gaf, stretch, st_scale
import torch

class ForexGafData(Dataset):
    def __init__(self, prices, seq_length):
        prices = np.array(prices)
        values = np.log(prices)
        values = values[:len(values) // seq_length * seq_length]
        tensor = torch.from_numpy(values).float().view(-1, seq_length)
        tensor = stretch(tensor)
        tensor = pt_gaf(tensor)

        # Remove matrices where nan is present
        tensor = tensor[~torch.isnan(tensor.view(tensor.size(0),-1)).any(axis=1)]
        self.x = tensor.view(-1, 1, seq_length, seq_length)

    def __len__(self):
        return self.x.size(0) 

    def __getitem__(self, idx):
        out = self.x[idx]
        return out
