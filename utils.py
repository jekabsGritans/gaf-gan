import numpy as np

def rolling(x,n):
    out = []
    for i in range(n,len(x)):
        out.append(np.mean(x[i-n:i]))
    return out


import torch
import torch.nn as nn

def gradient_penalty(critic, real, fake, device):
    BATCH_SIZE, DATA_SIZE = real.shape
    epsilon = torch.rand((BATCH_SIZE, 1)).repeat(1, DATA_SIZE).to(device)
    interpolated_data = real * epsilon + fake * (1-epsilon)
    
    mixed_scores = critic(interpolated_data)

    gradient = torch.autograd.grad(
        inputs=interpolated_data,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]

    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    return torch.mean((gradient_norm-1) ** 2)


import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

def save_progression(static_samples):
    fig, axes = plt.subplots(3,3,figsize=(9, 9))

    ims = []
    def plot(samples, axes):
        for sample, ax in zip(samples, axes.flatten()):
            ims.append(ax.imshow(np.squeeze(sample)))#, cmap='gray_r'))
            ax.axis('off')

    def update(samples, axes):
        for sample, im in zip(samples,ims):
            im.set_data(np.squeeze(sample))

    plot(static_samples[0], axes)
    def animate(i):
        return update(static_samples[i], axes)

    anim = FuncAnimation(fig, animate, interval=100, frames=len(static_samples)-1)

    writervideo = FFMpegWriter(fps=120)
    anim.save('progress.mp4', writer=writervideo)
    plt.close()