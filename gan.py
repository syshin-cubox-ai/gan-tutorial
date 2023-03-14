from typing import Tuple

import numpy as np
import torch.backends.cudnn
import torch.nn as nn
import torch.utils.data

import common


class GAN(common.BaseGAN):
    def __init__(self, config: dict, generator: nn.Module, discriminator: nn.Module, z_dim: int):
        super().__init__(config, generator, discriminator, z_dim)

    def train_step(
            self,
            imgs: torch.Tensor,
            labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        imgs = imgs.to(self.device)
        real_labels = torch.ones((imgs.shape[0], 1), device=self.device)
        fake_labels = torch.zeros((imgs.shape[0], 1), device=self.device)

        # 1. Update discriminator: maximize log(D(x)) + log(1 - D(G(z))
        # 판별자가 real 이미지를 real로 인식하는 loss 계산
        self.d_optimizer.zero_grad(set_to_none=True)
        real_score = self.discriminator(imgs)
        d_loss_real = self.criterion(real_score, real_labels)

        # 랜덤 텐서로 fake 이미지를 생성하여 fake로 인식하는 loss 계산
        z = torch.randn((imgs.shape[0], self.z_dim), device=self.device)
        fake_img = self.generator(z)
        fake_score = self.discriminator(fake_img)
        d_loss_fake = self.criterion(fake_score, fake_labels)

        # real과 fake 이미지로 낸 오차를 더해서 최종 판별자 loss를 계산하고, 판별자 모델 업데이트
        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        self.d_optimizer.step()

        # 2. Update discriminator: maximize log(D(G(z)))
        # 생성자가 판별자를 속였는지에 대한 loss를 계산하고, 생성자 모델 가중치 업데이트
        self.g_optimizer.zero_grad(set_to_none=True)
        deception_score = self.discriminator(self.generator(z))
        g_loss = self.criterion(deception_score, real_labels)
        g_loss.backward()
        self.g_optimizer.step()

        return d_loss, g_loss, real_score, fake_score

    def infer(self, num_rows: int) -> np.ndarray:
        self.generator.eval()

        z = torch.randn((num_rows * 10, self.z_dim), device=self.device)
        with torch.no_grad():
            sample_img = self.generator(z)
        sample_img = self.post_process_sample_img(sample_img)
        return sample_img


if __name__ == '__main__':
    config = {
        'model_name': 'GAN',
        'batch_size': 256,
        'epoch': 500,
        'lr': 0.0002,
        'reproducibility': True,
        'num_workers': 8,
        'pin_memory': True,
        'prefetch_factor': 60000,
        'persistent_workers': True,
    }

    # Model
    generator = nn.Sequential(
        nn.Linear(64, 256),
        nn.ReLU(),
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Linear(256, 784),
        nn.Tanh(),
    )
    discriminator = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 256),
        nn.LeakyReLU(0.2),
        nn.Linear(256, 256),
        nn.LeakyReLU(0.2),
        nn.Linear(256, 1),
        nn.Sigmoid(),
    )

    gan = GAN(config, generator, discriminator, 64)
    if gan.select_training_or_demo() == 'Train':
        gan.train()
    else:
        gan.demo()
