from torch.utils.data import Dataset
from .encoders import Encoder, NegGasfEncoder, SimpleRasterizeEncoder, GasfEncoder
import numpy as np
import pandas as pd
import torch

class EncodedForexData(Dataset):
    def __init__(self, prices, seq_length, encoder: Encoder, relative=False, epsilon=0):
        prices = np.array(prices)
        values = np.log(prices)
        if relative:
            values = np.diff(values)

        values = values[:len(values)//seq_length*seq_length]
        tensor = torch.from_numpy(values).float().view(-1,seq_length)

        if relative:
            tensor /= (tensor.abs().max(dim=1).values.view(-1,1)+epsilon)
        else:
            tensor -= tensor.min(dim=1).values.view(-1,1)
            tensor /= (tensor.max(dim=1).values.view(-1,1)+epsilon)

        tensor = encoder.encode(tensor)

        # Remove matrices where nan is present
        tensor = tensor[~torch.isnan(tensor.view(tensor.size(0),-1)).any(axis=1)]
        self.x = tensor 

    def __len__(self):
        return self.x.size(0) 

    def __getitem__(self, idx):
        out = self.x[idx]
        return out

def get_dataset(method, path='data/erususd_minute.csv'):
    SEQ_LENGTH=64
    prices = pd.read_csv(path)['BidClose'].values
    encoder = {
        'simple': SimpleRasterizeEncoder(),
        'relative': SimpleRasterizeEncoder(),
        'gasf': GasfEncoder(),
        'relative_gasf': NegGasfEncoder(),
    }[method]
    return EncodedForexData(prices, SEQ_LENGTH, encoder, relative='relative' in method, epsilon=1e-6 if 'gasf' in method else 0)