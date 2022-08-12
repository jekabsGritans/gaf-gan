import numpy as np
import torch

def pt_gaf(x):
    x = torch.arccos(x)
    x = x.reshape(-1,1)+x
    x = torch.cos(x)
    return x

class GAFTransform:
    def __call__(self, x):
        x = np.arccos(x)
        x = x.reshape(-1, 1)+x
        x = np.cos(x)
        return x
    
    @staticmethod
    def reverse(mat):
        n = mat.shape[0]
        mat = np.arccos(mat)
        l = [(mat[0,1]+mat[0,2]-mat[1,2])/2]
        for i in range(1,n):
            l.append(mat[i-1,i] - l[-1])
        arr = np.array(l)
        return np.cos(arr)

def scale(x):
    return x/x.abs().max()

def stretch(x):
    x = x-x.min()
    x = x/(x.max()-x.min())
    return x

