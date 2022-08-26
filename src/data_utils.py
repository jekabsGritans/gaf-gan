import numpy as np
import torch

import matplotlib.pyplot as plt
def plot_cdf(data, ax=None, **kwargs):
    y = 1. * np.arange(len(data)) / (len(data) - 1)
    if ax is None:
        ax = plt.gca()
    ax.plot(np.sort(data), y, **kwargs)

def get_gp(x_real, x_fake, critic, device):
    batch_size = x_real.size()[0]
    alpha = torch.rand(batch_size, 1).view(batch_size,1,1,1)
    alpha = alpha.expand(x_real.size()).to(device)
    interpolates = alpha * x_real + ((1 - alpha) * x_fake)
    interpolates = interpolates.to(device)
    interpolates.requires_grad_(True)
    
    pred_interpolates = critic(interpolates)

    # Calculate gradients of critic estimations wrt interpolates
    gradients = torch.autograd.grad(outputs=pred_interpolates, inputs=interpolates,
                                    grad_outputs=torch.ones(pred_interpolates.size()).to(device),
                                    create_graph=True, retain_graph=True, only_inputs=True)[0]
    
    # gradient_penalty = ((gradients_norm - 1) ** 2).mean() * self.gp_weight
    # gradient_norm = gradients.view(gradients.size(0), -1).norm(2, dim=1)
    # gradient_penalty = torch.mean((gradient_norm - 1) ** 2) * self.gp_weight

    gradient_penalty = torch.mean((1. - torch.sqrt(1e-12+torch.sum(gradients.view(gradients.size(0), -1)**2, dim=1)))**2)

    return gradient_penalty


def get_model_data(model, decoder, n, device):
    sample_batches = []
    with torch.no_grad():
        noise = torch.randn(n,100).to(device)
        for i in range(n//100):
            sample_batches.append(model(noise[i*100:(i+1)*100]))

    samples = []
    for batch in sample_batches:
        samples.append(np.concatenate(decoder(batch)))

    return np.concatenate(samples)


from scipy.stats import kstest
def get_ks(model, decoder, data, device, n=1000):
    while True:
        model_data = get_model_data(model, decoder, n, device)
        ks = kstest(data, model_data)

        if ks[1] < 0.05:
            break
        n = n*2
            
            
    return ks.statistic
