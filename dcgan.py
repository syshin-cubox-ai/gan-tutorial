from typing import Tuple

import torch
import torch.nn as nn

import gan


class DCGAN(gan.GAN):
    def __init__(
            self,
            config: dict,
            generator: nn.Module,
            discriminator: nn.Module,
    ):
        super().__init__(config, generator, discriminator)

    def train_step(
            self,
            imgs: torch.Tensor,
            labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        imgs = imgs.to(self.device)
        real_labels = torch.ones((imgs.shape[0], 1), device=self.device)
        fake_labels = torch.zeros((imgs.shape[0], 1), device=self.device)

        # 판별자가 real 이미지를 real로 인식하는 loss 계산
        real_score = self.discriminator(imgs)
        d_loss_real = self.criterion(real_score, real_labels)

        # 랜덤 텐서로 fake 이미지 생성
        z = torch.randn((imgs.shape[0], 100), device=self.device)
        sample_img = self.generator(z)

        # 판별자가 fake 이미지를 fake로 인식하는 loss 계산
        fake_score = self.discriminator(sample_img)
        d_loss_fake = self.criterion(fake_score, fake_labels)

        # real과 fake 이미지로 낸 오차를 더해서 최종 판별자 loss로 계산
        d_loss = d_loss_real + d_loss_fake

        # 판별자 모델 가중치 업데이트
        self.d_optimizer.zero_grad(set_to_none=True)
        d_loss.backward()
        self.d_optimizer.step()

        # 생성자가 판별자를 속였는지에 대한 loss 계산
        deception_score = self.discriminator(self.generator(z))
        g_loss = self.criterion(deception_score, real_labels)

        # 생성자 모델 가중치 업데이트
        self.g_optimizer.zero_grad(set_to_none=True)
        g_loss.backward()
        self.g_optimizer.step()

        return d_loss, g_loss, real_score, fake_score


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

    gan = DCGAN(config, generator, discriminator)
    if gan.select_training_or_demo() == 'Train':
        gan.train()
    else:
        gan.demo()
