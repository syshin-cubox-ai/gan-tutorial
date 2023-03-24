import torch
import torch.nn as nn

import gan


# Generator와 Discriminator에 적용시킬 커스텀 가중치 초기화 함수
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Generator(nn.Module):
    def __init__(self, nc=1, nz=100, ngf=64):
        super().__init__()
        self.main = nn.Sequential(
            # 입력데이터 Z가 가장 처음 통과하는 전치 합성곱 계층입니다. (Project and reshape)
            nn.Unflatten(1, (nz, 1, 1)),
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(),
            # 위의 계층을 통과한 데이터의 크기. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(),
            # 위의 계층을 통과한 데이터의 크기. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(),
            # 위의 계층을 통과한 데이터의 크기. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(),
            # 위의 계층을 통과한 데이터의 크기. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 1, 1, 2, bias=False),
            nn.Tanh(),
            # 위의 계층을 통과한 데이터의 크기. (nc) x 28 x 28
        )

    def forward(self, x):
        return self.main(x)


class Discriminator(nn.Module):
    def __init__(self, nc=1, ndf=64):
        super().__init__()
        self.main = nn.Sequential(
            # 입력 데이터의 크기는 (nc) x 28 x 28 입니다
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2),
            # 위의 계층을 통과한 데이터의 크기. (ndf) x 14 x 14
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2),
            # 위의 계층을 통과한 데이터의 크기. (ndf*2) x 7 x 7
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2),
            # 위의 계층을 통과한 데이터의 크기. (ndf*4) x 3 x 3
            nn.Conv2d(ndf * 4, 1, 4, 2, 1, bias=False),
            nn.Sigmoid(),
            # 위의 계층을 통과한 데이터의 크기. (nc) x 1 x 1
            nn.Flatten(),
            # 위의 계층을 통과한 데이터의 크기. (nc)
        )

    def forward(self, x):
        return self.main(x)


class DCGAN(gan.GAN):
    def __init__(
            self,
            config: dict,
            generator: nn.Module,
            discriminator: nn.Module,
            z_dim: int,
    ):
        super().__init__(config, generator, discriminator, z_dim)

    def build_optimizer(self):
        self.g_optimizer = torch.optim.RAdam(self.generator.parameters(), self.config['lr'], (0.5, 0.999))
        self.d_optimizer = torch.optim.RAdam(self.discriminator.parameters(), self.config['lr'], (0.5, 0.999))


if __name__ == '__main__':
    config = {
        'model_name': 'DCGAN',
        'batch_size': 256,
        'epoch': 50,
        'lr': 0.0002,
        'num_workers': 8,
        'pin_memory': True,
        'prefetch_factor': 60000,
        'persistent_workers': True,
    }

    # Model
    generator = Generator()
    discriminator = Discriminator()
    generator.apply(weights_init)
    discriminator.apply(weights_init)

    dcgan = DCGAN(config, generator, discriminator, 100)
    if dcgan.select_training_or_demo() == 'Train':
        dcgan.train()
    else:
        dcgan.demo()
