import torch

def pt_gaf(x):
    x = torch.arccos(x)
    x = x.view(-1,1,x.size(1))+x.view(-1,x.size(1),1)
    x = torch.cos(x)
    return x


def reverse_gaf(mats):
    vals = torch.diagonal(mats, dim1=1, dim2=2)
    out = torch.cos((torch.arccos(vals)/2)).squeeze()
    return out

# Scale to [0,1]
def stretch(series):
    maxes = torch.max(series, dim=1).values.view(-1, 1)
    mins = torch.min(series, dim=1).values.view(-1,1)
    out = (series-mins)/(maxes-mins)
    return out


def reverse_noisy_gaf(mat):
    a = mat.shape[0]
    S = mat.sum()/(2*a)
    vals = torch.zeros(a)
    for n in range(mat.shape[0]):
        vals[n] = (mat[n,:].sum()+mat[:,n].sum()-2*S)/(2*a)
    return vals

# Scale to [-1,1]
def st_scale(series):
    return series / series.abs().max(dim=1).values.view(-1, 1)

