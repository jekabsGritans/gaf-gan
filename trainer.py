from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
from utils import rolling
import numpy as np
from abc import ABC, abstractmethod, abstractstaticmethod

class GanTrainer(ABC):
    def __init__(self, generator, critic, gen_optimizer, critic_optimizer,
                latent_dimension, device):
        self.g = generator
        self.c = critic
        self.g_optim = gen_optimizer
        self.c_optim = critic_optimizer
        self.latent_dim = latent_dimension
        self.device = device

        if self.device:
            self.c.to(self.device)
            self.g.to(self.device)
        
        self.losses = {'g': [], 'c': []}

        self._curr_epoch = 0
        self._epochs = 0

        latent_shape = (1, self.latent_dim)
        self._static_noise = self.sample_latent(latent_shape)   
        if self.device:
            self._static_noise = self._static_noise.to(self.device)

        self.static_samples = []

    def save_sample(self):
        sample = self.g(self._static_noise).detach().cpu().numpy()
        self.static_samples.append(sample)

    @abstractmethod
    def train_critic_iteration(self, x_real):
        pass
    
    @abstractmethod
    def train_gen_iteration(self, x_real):
        pass

    @abstractmethod
    def train_epoch(self, data_loader):
        pass

    def epoch_wrapper(self, data_loader):
        status_bar = tqdm(data_loader)
        status_bar.set_description(f'Epoch [{self._curr_epoch}/{self._epochs}]')
        return status_bar

    def train(self, data_loader, epochs):
        self.save_sample()
        print(f"Training...")
        self._epochs = epochs
        for epoch in range(epochs):
            self._curr_epoch = epoch+1
            self.train_epoch(data_loader)
        
    def sample_generator(self, n):
        latent_shape = (n, self.latent_dim)
        noise = self.sample_latent(latent_shape)
        if self.device:
            noise = noise.to(self.device)
        return self.g(noise)

    def sample(self, n):
        latent_shape = (n, self.latent_dim)
        noise = self.sample_latent(latent_shape)
        if self.device:
            noise = noise.to(self.device)
        return self.g(noise).detach().cpu().numpy()

    def plot_losses(self):
        plt.plot(self.losses['g'], label='Generator')
        plt.plot(self.losses['c'], label='Critic')
        plt.legend()
        plt.show()
    
    @staticmethod
    def sample_latent(shape):
        sample = torch.randn(shape)
        return sample


class WGanTrainer(GanTrainer):
    def __init__(self, generator, critic, gen_optimizer, critic_optimizer,
                latent_dimension, device, critic_iterations=5, weigth_clip=0.01):
        super().__init__(generator, critic, gen_optimizer, critic_optimizer,
                latent_dimension, device)

        self.__step_num = 0
        self.weight_clip = weigth_clip
        self.critic_iterations = critic_iterations
        
    def train_epoch(self, data_loader):
        for x_real in self.epoch_wrapper(data_loader):
            if isinstance(x_real, (list, tuple)):
                x_real = x_real[0]
            x_real = x_real.float()
            if self.device:
                x_real = x_real.to(self.device)

            self.train_critic_iteration(x_real)
            if self.__step_num % self.critic_iterations == 0:
                self.train_gen_iteration(x_real)
            
            if self.__step_num % 10 == 0:
                self.save_sample()
                
            self.__step_num += 1

    def train_critic_iteration(self, x_real):
        batch_size = x_real.size(0)
        x_fake = self.sample_generator(batch_size)
        if self.device:
            x_real = x_real.to(self.device)
        
        c_real = self.c(x_real)
        c_fake = self.c(x_fake)
        loss_critic = -(torch.mean(c_real) - torch.mean(c_fake))

        self.c_optim.zero_grad()
        loss_critic.backward()
        self.c_optim.step()

        for p in self.c.parameters():
            p.data.clamp_(-self.weight_clip, self.weight_clip)

        self.losses['c'].append(loss_critic.item())
    
    def train_gen_iteration(self, x_real):
        batch_size = x_real.size(0)
        x_fake = self.sample_generator(batch_size)
        c_fake = self.c(x_fake)
        loss_gen = -torch.mean(c_fake)

        self.g_optim.zero_grad()
        loss_gen.backward()
        self.g_optim.step()

        self.losses['g'].append(loss_gen.item())
    
    def plot_losses(self):
        plt.plot(rolling(self.losses['g'], 10), label='Generator')
        plt.plot(rolling(self.losses['c'][::self.critic_iterations], 10), label='Critic')
        plt.legend()
        plt.show()


class WGanGpTrainer(WGanTrainer):
    def __init__(self, generator, critic, gen_optimizer, critic_optimizer,
                latent_dimension, device=None, critic_iterations=5, gp_weight=10):
        super().__init__(generator, critic, gen_optimizer, critic_optimizer,
                latent_dimension, device, critic_iterations=critic_iterations)
        
        self.gp_weight = gp_weight
        self.losses = {'g': [], 'c': [], 'gp': []}


    def _gradient_penalty(self, x_real, x_fake):
        batch_size = x_real.size()[0]
        alpha = torch.rand(batch_size, 1)
        alpha = alpha.expand_as(x_real)
        if self.device:
            alpha = alpha.to(self.device)
        
        interpolates = alpha * x_real + (1 - alpha) * x_fake
        interpolates.requires_grad_(True)
        if self.device:
            interpolates = interpolates.to(self.device)

        pred_interpolates = self.c(interpolates)

        # Calculate gradients of critic estimations wrt interpolates
        gradients = torch.autograd.grad(outputs=pred_interpolates, inputs=interpolates,
                                        grad_outputs=torch.ones(pred_interpolates.size()).to(self.device) if self.device else torch.ones(pred_interpolates.size()),
                                        create_graph=True, retain_graph=True)[0]
        
        gradients = gradients.view(batch_size, -1)

        # Add epsilon to prevent nans
        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)
        
        gradient_penalty = ((gradients_norm - 1) ** 2).mean() * self.gp_weight
        return gradient_penalty

    def train_critic_iteration(self, x_real):
        batch_size = x_real.size(0)
        x_fake = self.sample_generator(batch_size)

        if self.device:
            x_real = x_real.to(self.device)
        
        self.c_optim.zero_grad()

        c_real = self.c(x_real)
        c_fake = self.c(x_fake)

        gp = self._gradient_penalty(x_real, x_fake)
        critic_loss = -(torch.mean(c_real) - torch.mean(c_fake)) + gp
        critic_loss.backward()
        self.c_optim.step()
        
        self.losses['c'].append(critic_loss.item())
        self.losses['gp'].append(gp.item())

    def train_gen_iteration(self, x_real):
        batch_size = x_real.size(0)
        x_fake = self.sample_generator(batch_size)

        if self.device:
            x_real = x_real.to(self.device)
        
        self.g_optim.zero_grad()

        c_fake = self.c(x_fake).view(-1)

        gen_loss = -torch.mean(c_fake)
        gen_loss.backward()
        self.g_optim.step()
        
        self.losses['g'].append(gen_loss.item())

    def plot_losses(self):
        plt.plot(rolling(self.losses['g'], 10), label='Generator')

        c_wo_gp = np.array(self.losses['c']) - np.array(self.losses['gp'])
        plt.plot(rolling(c_wo_gp[::self.critic_iterations], 10), label='Critic')
        plt.legend()
        plt.show() 

class depWGanGpTrainer:

    def __init__(self, generator, critic, gen_optimizer, critic_optimizer,
                noise_size, gp_weight=10, critic_iterations=5, device=None):
        self.g = generator
        self.c = critic
        self.g_optim = gen_optimizer
        self.c_optim = critic_optimizer
        self.losses = {'g': [], 'c': [], 'GP': [], 'gradient_norm': []}
        self.device = device
        self.gp_weight = gp_weight
        self.critic_iterations = critic_iterations
        self.noise_size = noise_size

        self.num_steps = 0

        if self.device:
            self.g.to(self.device)
            self.c.to(self.device)

    def _critic_train_iteration(self, x_real):
        batch_size = x_real.shape[0]
        x_fake = self._sample_generator(batch_size)

        if self.device:
            x_real = x_real.to(self.device)

        self.c_optim.zero_grad()
        # Critic estimations
        pred_real = self.c(x_real)
        pred_fake = self.c(x_fake)

        # Gradient penalty
        gp = self._gradient_penalty(x_real, x_fake)
        # gp = gradient_penalty(self.c, x_real, x_fake)
        self.losses['GP'].append(gp.item())

        # Critic loss
        critic_loss = pred_fake.mean() - pred_real.mean() + gp # Inverted, because critic is maximising
        # critic_loss = -(torch.mean(critic_real) - torch.mean(critic_fake)) + gp*self.gp_weight

        critic_loss.backward()
        self.c_optim.step()

        self.losses['c'].append(critic_loss.item())

    def _generator_train_iteration(self, x_real):
        batch_size = x_real.shape[0]
        x_fake = self._sample_generator(batch_size)

        # Critic estimations
        pred_fake = self.c(x_fake)

        # Generator loss
        self.g_optim.zero_grad()
        gen_loss = -pred_fake.mean()
        gen_loss.backward()
        self.g_optim.step()

        self.losses['g'].append(gen_loss.item())

    def _gradient_penalty(self, x_real, x_fake):
        batch_size = x_real.size()[0]
        alpha = torch.rand(batch_size, 1)
        alpha = alpha.expand_as(x_real)
        if self.device:
            alpha = alpha.to(self.device)
        
        interpolates = alpha * x_real + (1 - alpha) * x_fake
        interpolates.requires_grad_(True)
        if self.device:
            interpolates = interpolates.to(self.device)

        pred_interpolates = self.c(interpolates)

        # Calculate gradients of critic estimations wrt interpolates
        gradients = torch.autograd.grad(outputs=pred_interpolates, inputs=interpolates,
                                        grad_outputs=torch.ones(pred_interpolates.size()).to(self.device) if self.device else torch.ones(pred_interpolates.size()),
                                        create_graph=True, retain_graph=True)[0]
        
        gradients = gradients.view(batch_size, -1)
        self.losses['gradient_norm'].append(gradients.norm(2, dim=1).mean().item())

        # Add epsilon to prevent nans
        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)
        
        gradient_penalty = ((gradients_norm - 1) ** 2).mean() * self.gp_weight
        return gradient_penalty


    def _train_epoch(self, data_loader, epoch):
        for x_real in tqdm(data_loader):
            x_real = x_real.float()
            self._critic_train_iteration(x_real)
            self.num_steps += 1
            # Instead of training critic on same batch few iterations, only train generator every few iterations
            if self.num_steps % self.critic_iterations == 0:
                self._generator_train_iteration(x_real)

    def train(self, data_loader, epochs):
        for epoch in range(epochs):
            self._train_epoch(data_loader, epoch+1)

    def plot_losses(self):
        self._plot(self.losses['g'], label='Generator')
        self._plot(self.losses['c'][::self.critic_iterations], label='Critic')
        plt.legend()
        plt.show()

        # self._plot(self.losses['GP'], label='GP')
        # self._plot(self.losses['gradient_norm'], label='Gradient norm')
        # plt.legend()
        # plt.show()

    def plot_samples(self, n=3):
        for data in self.sample(n):
            plt.plot(data)
        plt.show()
        
    def _plot(self, data, *args, **kwargs):
        plt.plot(rolling(data,1), *args, **kwargs)

    def _sample_generator(self, n):
        latent_shape = (n, self.noise_size)
        latent_samples = self.sample_latent(latent_shape)
        latent_samples.requires_grad_(True)
        if self.device:
            latent_samples = latent_samples.to(self.device)

        return self.g(latent_samples)
    
    def sample(self, n):
        latent_shape = (n, self.noise_size)
        latent_samples = self.sample_latent(latent_shape)
        return self.g(latent_samples).data.cpu().numpy()

    def sample_latent(self, shape):
        sample = torch.randn(shape)
        if self.device:
            sample = sample.to(self.device)
        return sample