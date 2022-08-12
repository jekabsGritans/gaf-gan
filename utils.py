import numpy as np

def rolling(x,n):
    out = []
    for i in range(n,len(x)):
        out.append(np.mean(x[i-n:i]))
    return out