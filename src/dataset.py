from torch.utils.data import Dataset
from .encoders import Encoder, SimpleRasterizeEncoder, GasfEncoder
import numpy as np
import pandas as pd
import torch

class EncodedForexData(Dataset):
    def __init__(self, prices, seq_length, encoder: Encoder, relative=False):
        prices = np.array(prices)
        values = np.log(prices)
        if relative:
            values = np.diff(values)

        values = values[:len(values)//seq_length*seq_length]
        tensor = torch.from_numpy(values).float().view(-1,seq_length)

        if relative:
            tensor /= tensor.abs().max(dim=1).values.view(-1,1)
        else:
            tensor -= tensor.min(dim=1).values.view(-1,1)
            tensor /= tensor.max(dim=1).values.view(-1,1)

        tensor = encoder.encode(tensor)

        # Remove matrices where nan is present
        tensor = tensor[~torch.isnan(tensor.view(tensor.size(0),-1)).any(axis=1)]
        self.x = tensor 

    def __len__(self):
        return self.x.size(0) 

    def __getitem__(self, idx):
        out = self.x[idx]
        return out

def get_dataset(method):
    SEQ_LENGTH=64
    prices = pd.read_csv('data/eurusd_minute.csv')['BidClose'].values
    encoder = {
        'simple': SimpleRasterizeEncoder(),
        'rel': SimpleRasterizeEncoder(),
        'gasf': GasfEncoder(),
    }[method]
    return EncodedForexData(prices, SEQ_LENGTH, encoder, relative=method=='rel')