from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
from utils import rolling
import numpy as np
from abc import ABC, abstractmethod, abstractstaticmethod

class GanTrainer(ABC):
    def __init__(self, generator, critic, gen_optimizer, critic_optimizer,
                latent_dimension, device, static_samples=9, epoch_callback=None):
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

        self._n_static_samples = static_samples
        latent_shape = (self._n_static_samples, self.latent_dim)
        self._static_noise = self.sample_latent(latent_shape)   
        if self.device:
            self._static_noise = self._static_noise.to(self.device)

        self.static_samples = []

        self.epoch_callback = epoch_callback

        self._step_num = 0

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
        self._step_num=0
        self.save_sample()
        print(f"Training...")
        self._epochs = epochs
        for epoch in range(epochs):
            self._curr_epoch = epoch+1
            self.train_epoch(data_loader)
            if self.epoch_callback:
                self.epoch_callback(self)
        
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


class ClassicalGanTrainer(GanTrainer):
    def __init__(self, generator, critic, gen_optimizer, critic_optimizer,
                latent_dimension, device, static_samples, epoch_callback):
    
        super().__init__(generator, critic, gen_optimizer, critic_optimizer,
                latent_dimension, device, static_samples, epoch_callback)

    
    def train_critic_iteration(self, x_real):
        if isinstance(x_real, (list, tuple)):
            x_real = x_real[0]

        if self.device:
            x_real = x_real.to(self.device)

        x_fake = self.sample_generator(x_real.size(0))

        loss = -(torch.log(self.c(x_real)).mean() + torch.log(1 - self.c(x_fake)).mean())

        self.c_optim.zero_grad()
        loss.backward()
        self.c_optim.step()
        self.losses['c'].append(loss.item())
    
    def train_gen_iteration(self, x_real):
        if isinstance(x_real, (list, tuple)):
            x_real = x_real[0]
        batch_size = x_real.size(0)
        x_fake = self.sample_generator(batch_size)

        loss = -torch.log(self.c(x_fake)).mean()
        self.g_optim.zero_grad()
        loss.backward()
        self.g_optim.step()
        self.losses['g'].append(loss.item())

    def train_epoch(self, data_loader):
        for x_real in self.epoch_wrapper(data_loader):
            self.train_critic_iteration(x_real)
            self.train_gen_iteration(x_real)

            if self._step_num % 10 == 0:
                self.save_sample()
            
            self._step_num += 1

class WGanTrainer(GanTrainer):
    def __init__(self, generator, critic, gen_optimizer, critic_optimizer,
                latent_dimension, device, static_samples=9, epoch_callback=None,
            critic_iterations=5,
            weigth_clip=0.01
            ):

        super().__init__(
                generator=generator,
                critic=critic,
                gen_optimizer=gen_optimizer,
                critic_optimizer=critic_optimizer,
                latent_dimension=latent_dimension,
                device=device,
                static_samples=static_samples,
                epoch_callback=epoch_callback,
                )
        self._step_num=0

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
            if self._step_num % self.critic_iterations == 0:
                self.train_gen_iteration(x_real)
            
            if self._step_num % 10 == 0:
                self.save_sample()
                
            self._step_num += 1

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
                latent_dimension, device=None, static_samples=9, epoch_callback=None,
                critic_iterations=5,
            gp_weight=10):

        super().__init__(
            generator=generator,
            critic=critic,
            gen_optimizer=gen_optimizer,
            critic_optimizer=critic_optimizer,
            latent_dimension=latent_dimension,
            device=device,
            static_samples=static_samples,
            epoch_callback=epoch_callback,
            critic_iterations=critic_iterations,
            )
        
        self.gp_weight = gp_weight
        self.losses = {'g': [], 'c': [], 'gp': []}

    def _gradient_penalty(self, x_real, x_fake):
        batch_size = x_real.size()[0]
        alpha = torch.rand(batch_size, 1).view(batch_size,1,1,1)
        alpha = alpha.expand(x_real.size())
        if self.device:
            alpha = alpha.to(self.device)
        
        interpolates = alpha * x_real + ((1 - alpha) * x_fake)
        interpolates.requires_grad_(True)
        if self.device:
            interpolates = interpolates.to(self.device)

        pred_interpolates = self.c(interpolates)

        # Calculate gradients of critic estimations wrt interpolates
        gradients = torch.autograd.grad(outputs=pred_interpolates, inputs=interpolates,
                                        grad_outputs=torch.ones(pred_interpolates.size()).to(self.device) if self.device else torch.ones(pred_interpolates.size()),
                                        create_graph=True, retain_graph=True, only_inputs=True)[0]
        
        # gradient_penalty = ((gradients_norm - 1) ** 2).mean() * self.gp_weight
        
        gradient_penalty = torch.mean((1. - torch.sqrt(1e-8+torch.sum(gradients.view(gradients.size(0), -1)**2, dim=1)))**2)

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