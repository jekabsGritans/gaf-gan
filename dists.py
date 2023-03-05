import numpy as np
import matplotlib.pyplot as plt

def gasf(arr):
    theta = np.arccos(arr)
    theta = theta[:,None] + theta[None,:]
    return np.cos(theta)

arr = np.random.rand(32)
mat = gasf(arr)


# fig, axs = plt.subplots(1, 4)

# axs[0].hist(arr, bins=40)
# axs[1].hist(mat.flatten(), bins=40)

# plt.show()


def sig(a,b):
    a = (a+1)/2
    b = (b+1)/2
    return np.cos((np.arccos(a) + np.arccos(b)))

def tanh(a,b):
    return np.cos((np.arccos(a) + np.arccos(b)))


a = -0.5
b = 1

print(sig(a,b))
print(tanh(a,b))