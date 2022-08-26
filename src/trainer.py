from tqdm import tqdm
import torch
from .utils import  item
import numpy as np
from abc import ABC, abstractmethod
import os
from torch.utils.tensorboard import SummaryWriter
import torchvision


class GanTrainer(ABC):
    def __init__(self, generator, critic, gen_optimizer, critic_optimizer,
                latent_dimension, device, critic_iterations=1, static_samples=16,
                model_dir=None, write_dir=None, checkpoint=None,
                checkpoint_interval=0):
        
        self.g = generator
        self.c = critic
        self.critic_iterations = critic_iterations
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
        self._checkpoint_num = 0

    @abstractmethod
    def train_critic_iteration(self, x_real):
        pass
    
    @abstractmethod
    def train_gen_iteration(self, x_real):
        pass


    def train_epoch(self, data_loader):
        for batch in self.epoch_wrapper(data_loader):
            self.train_critic_iteration(batch)
            
            if self._step_num % self.critic_iterations == 0:
                self.train_gen_iteration(batch)

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
            self.display_samples(static_samples, 'Static Samples', self._step_num)

            # Show dynamic samples
            dynamic_samples = self.sample_generator(self._n_static_samples)
            self.display_samples(dynamic_samples, 'Dynamic Samples', self._step_num)

    def epoch_wrapper(self, data_loader):
        status_bar = tqdm(data_loader)
        status_bar.set_description(f'Epoch [{self._curr_epoch}/{self._epochs}]')
        return status_bar

    def display_samples(self, samples, title, num):
        display_samples = samples.detach().clone()
        grid = torchvision.utils.make_grid(display_samples)

        # Tensorboard only supports 1 or 3 channels
        if grid.size(0) == 2:
            zeros = torch.zeros(1, grid.size(1), grid.size(2))
            if self.device:
                zeros = zeros.to(self.device)
            grid = torch.cat((grid, zeros), dim=0)
        self.writer.add_image(title, grid, num)

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
            # Descrine training           
            desc = f"""
            Training {self.__class__.__name__}
            gen: {self.g.__class__.__name__}
            g_optim: {self.g_optim.__class__.__name__} (lr: {self.g_optim.param_groups[0]['lr']})
            c: {self.c.__class__.__name__}
            c_optim: {self.c_optim.__class__.__name__} (lr: {self.c_optim.param_groups[0]['lr']})

            epochs: {epochs}
            batch_size: {data_loader.batch_size}
            """
            self.writer.add_text('Training', desc, self._step_num)

            # Show dataset samples
            training_examples = item(next(iter(data_loader)))
            if self.device:
                training_examples = training_examples.to(self.device)

            self.display_samples(training_examples, 'Training Batch', 0)

            # Visualize model
            self.writer.add_graph(self.c, training_examples)

            # Show pre-training static samples
            self.display_samples(static_samples, 'Static Samples', 0)

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

    @staticmethod
    def sample_latent(shape):
        sample = torch.randn(shape)
        return sample
    
    def save_progress(self):
        if self.model_dir:
            print('Saving model checkpoint...')
            dirpath = os.path.join(self.model_dir, 'checkpoints', f'{self._checkpoint_num}')
            g_path = os.path.join(dirpath, 'g.pt')
            c_path = os.path.join(dirpath, 'c.pt')
            os.makedirs(dirpath, exist_ok=True)
            torch.save(self.g.state_dict(), g_path)
            torch.save(self.c.state_dict(), c_path)
            self._checkpoint_num += 1



class ClassicalGanTrainer(GanTrainer):
    def __init__(self, generator, critic, gen_optimizer, critic_optimizer,
                latent_dimension, device, critic_iterations=1, static_samples=16,
                model_dir=None, write_dir=None, checkpoint=None, checkpoint_interval=0):
    
        super().__init__(
                generator=generator,
                critic=critic,
                critic_iterations=critic_iterations,
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

        fake_pred = self.c(x_fake)
        real_pred = self.c(x_real)

        criterion = torch.nn.BCELoss()
        sones = torch.FloatTensor(x_real.size(0),1).uniform_(0.8,1.1)
        szeros = torch.FloatTensor(x_real.size(0),1).uniform_(0.0,0.3)
        if self.device:
            sones = sones.to(self.device)
            szeros = szeros.to(self.device)

        loss_real = criterion(real_pred, sones)
        loss_fake = criterion(fake_pred, szeros)
        loss_c = (loss_real + loss_fake) / 2

        self.c_optim.zero_grad()
        loss_c.backward(retain_graph=True)
        self.c_optim.step()

        # Train gen
        fake_pred= self.c(x_fake)
        sones = torch.FloatTensor(x_real.size(0),1).uniform_(0.8,1.1)
        if self.device:
            sones = sones.to(self.device)
        loss_g = criterion(fake_pred, sones)
        self.g_optim.zero_grad()
        loss_g.backward()
        self.g_optim.step()

        self.losses['c'].append(loss_c.item())
        self.losses['g'].append(loss_g.item())

    def train_gen_iteration(self, x_real):
        pass

class WGanTrainer(GanTrainer):
    def __init__(self, generator, critic, gen_optimizer, critic_optimizer,
                latent_dimension, device=None, critic_iterations=5,
                static_samples=16, model_dir=None, write_dir=None,
                checkpoint=None, checkpoint_interval=0,
            weigth_clip=0.01
            ):

        super().__init__(
                generator=generator,
                critic=critic,
                critic_iterations=critic_iterations,
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

        self.losses['c'].append(-loss_critic.item())
    
    def train_gen_iteration(self, x_real):
        batch_size = x_real.size(0)
        x_fake = self.sample_generator(batch_size)
        c_fake = self.c(x_fake)
        loss_gen = -torch.mean(c_fake)

        self.g_optim.zero_grad()
        loss_gen.backward()
        self.g_optim.step()

        self.losses['g'].append(-loss_gen.item())
    
    
class WGanGpTrainer(WGanTrainer):
    def __init__(self, generator, critic, gen_optimizer,
                critic_optimizer, latent_dimension, device=None,
                critic_iterations=5, static_samples=16, model_dir=None,
                write_dir=None, checkpoint=None, checkpoint_interval=None,
            gp_weight=10):

        super().__init__(
            generator=generator,
            critic=critic,
            critic_iterations=critic_iterations,
            gen_optimizer=gen_optimizer,
            critic_optimizer=critic_optimizer,
            latent_dimension=latent_dimension,
            device=device,
            static_samples=static_samples,
            model_dir=model_dir,
            write_dir=write_dir,
            checkpoint=checkpoint,
            checkpoint_interval=checkpoint_interval,
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
        # gradient_norm = gradients.view(gradients.size(0), -1).norm(2, dim=1)
        # gradient_penalty = torch.mean((gradient_norm - 1) ** 2) * self.gp_weight

        gradient_penalty = torch.mean((1. - torch.sqrt(1e-12+torch.sum(gradients.view(gradients.size(0), -1)**2, dim=1)))**2)

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
        critic_loss = -(torch.mean(c_real) - torch.mean(c_fake))*0.5 + gp*self.gp_weight
        critic_loss.backward()
        self.c_optim.step()
        
        self.losses['c'].append(-critic_loss.item())
        self.losses['gp'].append(gp.item())


    def update_stats(self):
        static_samples = self.get_static_samples()
        assert not torch.isnan(static_samples).any()
        self.static_samples.append(static_samples)

        if self.writer:
            self.writer.add_scalars('Losses', {
                'Generator': np.mean(self.losses['g'][-10:]),
                'Critic': np.mean(self.losses['c'][-10:]),
                'Gradient Penalty': np.mean(self.losses['gp'][-10:]),
            } , self._step_num)
            
            # Show static samples
            self.display_samples(static_samples, 'Static Samples', self._step_num)

            # Show dynamic samples
            dynamic_samples = self.sample_generator(self._n_static_samples)
            self.display_samples(dynamic_samples, 'Dynamic Samples', self._step_num)