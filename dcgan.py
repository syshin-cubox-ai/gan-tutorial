import torch.nn as nn

import gan


class DCGAN(gan.GAN):
    def __init__(
            self,
            config: dict,
            generator: nn.Module,
            discriminator: nn.Module,
            z_dim: int,
    ):
        super().__init__(config, generator, discriminator, z_dim)


if __name__ == '__main__':
    config = {
        'model_name': 'DCGAN',
        'batch_size': 256,
        'epoch': 50,
        'lr': 0.0001,
        'reproducibility': True,
        'num_workers': 8,
        'pin_memory': True,
        'prefetch_factor': 60000,
        'persistent_workers': True,
    }

    # Model
    generator = nn.Sequential(
        nn.Linear(100, 256 * 7 * 7, bias=False),
        nn.BatchNorm1d(256 * 7 * 7),
        nn.LeakyReLU(0.3),
        nn.Unflatten(1, (256, 7, 7)),
        nn.ConvTranspose2d(256, 128, 5, 1, padding=2, bias=False),
        nn.BatchNorm2d(128),
        nn.LeakyReLU(0.3),
        nn.ConvTranspose2d(128, 64, 5, 2, padding=2, output_padding=1, bias=False),
        nn.BatchNorm2d(64),
        nn.LeakyReLU(0.3),
        nn.ConvTranspose2d(64, 1, 5, 2, padding=2, output_padding=1, bias=False),
        nn.Tanh(),
    )
    discriminator = nn.Sequential(
        nn.Conv2d(1, 64, 5, 2, 2, bias=False),
        nn.BatchNorm2d(64),
        nn.LeakyReLU(0.3),
        nn.Dropout(0.3),
        nn.Conv2d(64, 128, 5, 2, 2, bias=False),
        nn.BatchNorm2d(128),
        nn.LeakyReLU(0.3),
        nn.Dropout(0.3),
        nn.Flatten(),
        nn.Linear(6272, 1),
        nn.Sigmoid(),
    )

    gan = DCGAN(config, generator, discriminator, 100)
    if gan.select_training_or_demo() == 'Train':
        gan.train()
    else:
        gan.demo()
