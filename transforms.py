import numpy as np
import torch

def pt_gaf(x):
    x = torch.arccos(x)
    x = x.reshape(-1,1)+x
    x = torch.cos(x)
    return x

def reverse_gaf(mat):
    idxs = np.diag_indices_from(mat)
    vals = mat[idxs]
    out = torch.cos((torch.arccos(vals)/2))
    return out

def stretch(series):
    return (series - series.min()) / (series.max() - series.min())