import torch

def gasf(x):
    x = torch.arccos(x)
    x = x.view(-1,1,x.size(1))+x.view(-1,x.size(1),1)
    x = torch.cos(x)
    return x

def gadf(x):
    x = torch.asin(x)
    x = x.view(-1,1,x.size(1))-x.view(-1,x.size(1),1)
    x = torch.sin(x)
    return x

def rel_gaf(x):
    c1 = gasf(x)
    c2 = gadf(x)
    return torch.stack((c1,c2),dim=1)

def simple_raster(x):
    x = x.view(-1,1,x.size(1))
    mat = x.repeat(1,x.size(2),1)
    return mat

def reverse_gasf(mats):
    vals = torch.diagonal(mats, dim1=0, dim2=1)
    out = torch.cos((torch.arccos(vals)/2)).squeeze()
    return out

def reverse_noisy_gasf(mat):
    a = mat.shape[0]
    S = mat.sum()/(2*a)
    vals = torch.zeros(a)
    for n in range(mat.shape[0]):
        vals[n] = (mat[n,:].sum()+mat[:,n].sum()-2*S)/(2*a)
    return vals