from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
from utils import rolling, item
import numpy as np
from abc import ABC, abstractmethod
import os
from torch.utils.tensorboard import SummaryWriter
import torchvision

class GanTrainer(ABC):
    def __init__(self, generator, critic, gen_optimizer, critic_optimizer,
                latent_dimension, device, static_samples=16, model_dir=None,
                write_dir=None, checkpoint=None, checkpoint_interval=0):
        self.g = generator
        self.c = critic
        self.checkpoint = checkpoint
        self.checkpoint_interval = checkpoint_interval
        self.writer = None
        self.write_dir = write_dir
        self.g_optim = gen_optimizer
        self.c_optim = critic_optimizer
        self.latent_dim = latent_dimension
        self.device = device
        self.model_dir = model_dir
        self.write_dir = write_dir
        self.losses = {'g': [], 'c': []}
        self._curr_epoch = 0
        self._epochs = 0
        self._n_static_samples = static_samples
        latent_shape = (self._n_static_samples, self.latent_dim)
        self._static_noise = self.sample_latent(latent_shape)   
        self.static_samples = []
        self._step_num = 0

    @abstractmethod
    def train_critic_iteration(self, x_real):
        pass
    
    @abstractmethod
    def train_gen_iteration(self, x_real):
        pass

    @abstractmethod
    def train_batch(self, x_real):
        pass

    def train_epoch(self, data_loader):
        for batch in self.epoch_wrapper(data_loader):
            self.train_batch(batch)
            
            if self._step_num % 10 == 0:
                self.update_stats()

            self._step_num += 1
    
    def update_stats(self):
        static_samples = self.get_static_samples()
        assert not torch.isnan(static_samples).any()
        self.static_samples.append(static_samples)

        if self.writer:
            self.writer.add_scalars('Losses', {
                'Generator': np.mean(self.losses['g'][-10:]),
                'Critic': np.mean(self.losses['c'][-10:])
            } , self._step_num)
            
            # Show static samples
            grid = torchvision.utils.make_grid(static_samples)
            self.writer.add_image('Static Samples', grid, self._step_num)

            # Show dynamic samples
            dynamic_samples = self.sample_generator(self._n_static_samples)
            grid = torchvision.utils.make_grid(dynamic_samples)
            self.writer.add_image('Dynamic Samples', grid, self._step_num)




    def epoch_wrapper(self, data_loader):
        status_bar = tqdm(data_loader)
        status_bar.set_description(f'Epoch [{self._curr_epoch}/{self._epochs}]')
        return status_bar

    def train(self, data_loader, epochs):

        if self.checkpoint:
            self.g.load_state_dict(torch.load(os.path.join(self.checkpoint, 'g.pt')))
            self.c.load_state_dict(torch.load(os.path.join(self.checkpoint, 'c.pt')))

        self.writer = SummaryWriter(self.write_dir) if self.write_dir else None

        if self.device:
            self.c.to(self.device)
            self.g.to(self.device)

            self._static_noise = self._static_noise.to(self.device)

        self._step_num=0
        
        static_samples = self.get_static_samples()
        self.static_samples.append(static_samples)

        if self.writer:
            
            if not self.checkpoint:
                self.writer.delete_all_tensorboard_events()
                
            # Show dataset samples
            training_examples = item(next(iter(data_loader)))
            if self.device:
                training_examples = training_examples.to(self.device)
            grid = torchvision.utils.make_grid(training_examples)
            self.writer.add_image('Training Batch', grid, 0)

            # Visualize model
            self.writer.add_graph(self.c, training_examples)

            # Show pre-training static samples
            grid = torchvision.utils.make_grid(static_samples)
            self.writer.add_image('Progress', grid, 0)

        print("Training...")
        self._epochs = epochs
        for epoch in range(epochs):
            self._curr_epoch = epoch+1
            self.train_epoch(data_loader)
            if self.checkpoint_interval and (epoch+1) % self.checkpoint_interval == 0:
                self.save_progress()

        self.save_progress()
        print("Training complete.")

        if self.writer:
            # Add video of static sample progression
            static_samples = torch.stack(self.static_samples, axis=0).transpose(1,0)
            self.writer.add_video('Static Samples', static_samples, 0, fps=30)

            self.writer.close()

        
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

    def get_static_samples(self):
        return self.g(self._static_noise).detach()

    def plot_losses(self):
        plt.plot(self.losses['g'], label='Generator')
        plt.plot(self.losses['c'], label='Critic')
        plt.legend()
        plt.show()
    
    @staticmethod
    def sample_latent(shape):
        sample = torch.randn(shape)
        return sample
    
    def save_progress(self):
        if self.model_dir:
            torch.save(self.g.state_dict(), f'{self.model_dir}/g.pt')
            torch.save(self.c.state_dict(), f'{self.model_dir}/c.pt')



class ClassicalGanTrainer(GanTrainer):
    def __init__(self, generator, critic, gen_optimizer, critic_optimizer,
                latent_dimension, device, static_samples=16, model_dir=None,
                write_dir=None, checkpoint=None, checkpoint_interval=0):
    
        super().__init__(
                generator=generator,
                critic=critic,
                gen_optimizer=gen_optimizer,
                critic_optimizer=critic_optimizer,
                latent_dimension=latent_dimension,
                device=device,
                static_samples=static_samples,
                model_dir=model_dir,
                write_dir=write_dir,
                checkpoint=checkpoint,
                checkpoint_interval=checkpoint_interval
                )

    
    def train_critic_iteration(self, x_real):
        x_real = item(x_real)
        if self.device:
            x_real = x_real.to(self.device)

        x_fake = self.sample_generator(x_real.size(0))

        loss = -(torch.log(self.c(x_real)).mean() + torch.log(1 - self.c(x_fake)).mean())

        self.c_optim.zero_grad()
        loss.backward()
        self.c_optim.step()
        self.losses['c'].append(loss.item())
    
    def train_gen_iteration(self, x_real):
        batch_size = item(x_real).size(0)
        x_fake = self.sample_generator(batch_size)

        loss = -torch.log(self.c(x_fake)).mean()
        self.g_optim.zero_grad()
        loss.backward()
        self.g_optim.step()
        self.losses['g'].append(loss.item())

    def train_batch(self, x_real):
        if self.device:
            x_real = x_real.to(self.device)

        self.train_critic_iteration(x_real)
        self.train_gen_iteration(x_real)


class WGanTrainer(GanTrainer):
    def __init__(self, generator, critic, gen_optimizer, critic_optimizer,
                latent_dimension, device, static_samples=16, model_dir=None,
                write_dir=None, checkpoint=None, checkpoint_interval=0,
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
                model_dir=model_dir,
                write_dir=write_dir,
                checkpoint=checkpoint,
                checkpoint_interval=checkpoint_interval
                )
        self._step_num=0

        self.weight_clip = weigth_clip
        self.critic_iterations = critic_iterations
        
    def train_batch(self, x_real):
        x_real = item(x_real).float()
        if self.device:
            x_real = x_real.to(self.device)

        self.train_critic_iteration(x_real)
        if self._step_num % self.critic_iterations == 0:
            self.train_gen_iteration(x_real)
        

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
                latent_dimension, device=None, static_samples=9,
                model_dir=None, write_dir=None, checkpoint=None,
                checkpoint_interval=None, critic_iterations=5,
            gp_weight=10):

        super().__init__(
            generator=generator,
            critic=critic,
            gen_optimizer=gen_optimizer,
            critic_optimizer=critic_optimizer,
            latent_dimension=latent_dimension,
            device=device,
            static_samples=static_samples,
            model_dir=model_dir,
            write_dir=write_dir,
            checkpoint=checkpoint,
            checkpoint_interval=checkpoint_interval,
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