import numpy as np

def rolling(x,n):
    out = []
    for i in range(n,len(x)):
        out.append(np.mean(x[i-n:i]))
    return out

def item(x):
    return x[0] if isinstance(x,(tuple,list)) else x