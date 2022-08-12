import torch.nn as nn

class Hyperparameters(object):
  def __init__(self, **kwargs):
    self.__dict__.update(kwargs)

hp = Hyperparameters(n_epochs=200,
                     batch_size=64,
                     lr=.0002,
                     b1=.5,
                     b2=0.999,
                     n_cpu=8,
                     latent_dim=100,
                     img_size=32,
                     channels=1,
                     sample_interval=400)


class Generator(nn.Module):
  def __init__(self):
    super(Generator, self).__init__()

    self.init_size = hp.img_size // 4
    self.l1 = nn.Sequential(nn.Linear(hp.latent_dim, 128 * self.init_size ** 2))

    self.conv_blocks = nn.Sequential(
      nn.BatchNorm2d(128),
      nn.Upsample(scale_factor=2),
      nn.Conv2d(128, 128, 3, stride=1, padding=1),
      nn.BatchNorm2d(128, 0.8),
      nn.LeakyReLU(0.2, inplace=True),
      nn.Upsample(scale_factor=2),
      nn.Conv2d(128, 64, 3, stride=1, padding=1),
      nn.BatchNorm2d(64, 0.8),
      nn.LeakyReLU(0.2, inplace=True),
      nn.Conv2d(64, hp.channels, 3, stride=1, padding=1),
      nn.Tanh(),
    )

  def forward(self, z):
    out = self.l1(z)
    out = out.view(out.shape[0], 128, self.init_size, self.init_size)
    img = self.conv_blocks(out)
    return img
 


class Discriminator(nn.Module):
  def __init__(self):
    super(Discriminator, self).__init__()

    def discriminator_block(in_filters, out_filters, bn=True):
      block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), 
                nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
      if bn:
          block.append(nn.BatchNorm2d(out_filters, 0.8))
      return block

    self.model = nn.Sequential(
      *discriminator_block(hp.channels, 16, bn=False),
      *discriminator_block(16, 32),
      *discriminator_block(32, 64),
      *discriminator_block(64, 128),
    )

    # The height and width of downsampled image
    ds_size = hp.img_size // 2 ** 4
    self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1), nn.Sigmoid())

  def forward(self, img):
    out = self.model(img)
    out = out.view(out.shape[0], -1)
    validity = self.adv_layer(out)

    return validity