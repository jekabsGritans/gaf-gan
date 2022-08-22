import torch
import numpy as np

def gasf(x):
    """Gramian Angular Summation Field"""
    x = torch.arccos(x)
    x = x.view(-1,1,x.size(1))+x.view(-1,x.size(1),1)
    x = torch.cos(x)
    return x.view(-1,1,x.size(1), x.size(2))

def gadf(x):
    """Gramian Angular Difference Field"""
    x = torch.asin(x)
    x = x.view(-1,1,x.size(1))-x.view(-1,x.size(1),1)
    x = torch.sin(x)
    return x.view(-1,1,x.size(1), x.size(2))

def rg_gaf(x):
    """Stacked GAFs (for [-1;1] injection)"""
    c1 = gasf(x)
    c2 = gadf(x)
    return torch.stack((c1,c2),dim=1).view(-1,2,x.size(1),x.size(1))

def simple_raster(x):
    """Broadcast row to fill matrix"""
    x = x.view(-1,1,x.size(1))
    mat = x.repeat(1,x.size(2),1)
    return mat.view(-1,1,mat.size(1), mat.size(2))

def reverse_rel_raster(mat):
    """Reverse raster with relative values"""
    dim = {3:1,4:2}[mat.dim()]
    rels = mat.mean(dim=dim).detach().numpy()
    return rels.squeeze()

def reverse_gasf(mat):
    """Reverse GASF with absolute values"""
    vals = torch.diagonal(mat, dim1=0, dim2=1)
    log_prices = torch.cos((torch.arccos(vals)/2)).squeeze().detach().numpy()
    rels = np.diff(log_prices)
    return rels

def reverse_noisy_gasf(mat):
    a = mat.shape[0]
    S = mat.sum()/(2*a)
    vals = torch.zeros(a)
    for n in range(mat.shape[0]):
        vals[n] = (mat[n,:].sum()+mat[:,n].sum()-2*S)/(2*a)
    return vals