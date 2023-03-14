import os
import random

import cv2
import numpy as np
import torch.backends.cudnn
import torch.nn as nn
import torch.utils.data
import torchvision
import torchvision.transforms.functional as F
import tqdm
import wandb
from PIL import Image


def train(
        epoch: int,
        trainloader: torch.utils.data.DataLoader,
        generator: nn.Module,
        discriminator: nn.Module,
        criterion: nn.Module,
        g_optimizer: torch.optim.Optimizer,
        d_optimizer: torch.optim.Optimizer,
        device: torch.device,
):
    d_loss = None
    g_loss = None
    real_score = None
    fake_score = None
    frames = []
    for eph in tqdm.tqdm(range(epoch), 'Epoch', position=0):
        for images, labels in tqdm.tqdm(trainloader, 'Batch', leave=False, position=1):
            images = torch.flatten(images, start_dim=1).to(device)
            real_labels = torch.ones((images.shape[0], 1), device=device)
            fake_labels = torch.zeros((images.shape[0], 1), device=device)

            # 판별자가 real 이미지를 real로 인식하는 loss 계산
            real_score = discriminator(images)
            d_loss_real = criterion(real_score, real_labels)

            # 랜덤 텐서로 fake 이미지 생성
            z = torch.randn((images.shape[0], 64), device=device)
            fake_images = generator(z)

            # 판별자가 fake 이미지를 fake로 인식하는 loss 계산
            fake_score = discriminator(fake_images)
            d_loss_fake = criterion(fake_score, fake_labels)

            # real과 fake 이미지로 낸 오차를 더해서 최종 판별자 loss로 계산
            d_loss = d_loss_real + d_loss_fake

            # 판별자 모델 가중치 업데이트
            d_optimizer.zero_grad(set_to_none=True)
            d_loss.backward()
            d_optimizer.step()

            # 생성자가 판별자를 속였는지에 대한 loss 계산
            deception_score = discriminator(generator(z))
            g_loss = criterion(deception_score, real_labels)

            # 생성자 모델 가중치 업데이트
            g_optimizer.zero_grad(set_to_none=True)
            g_loss.backward()
            g_optimizer.step()

        # Get training information
        d_loss = d_loss.item()
        g_loss = g_loss.item()
        real_score = real_score.mean().item()
        fake_score = fake_score.mean().item()

        # Get sample fake images
        fake_images = infer(generator, device, 40)
        frames.append(Image.fromarray(fake_images))

        # Log to wandb
        wandb.log({
            'd_loss': d_loss,
            'g_loss': g_loss,
            'D(x)': real_score,
            'D(G(z))': fake_score,
            'g_optimizer_lr': g_optimizer.param_groups[0]['lr'],
            'd_optimizer_lr': d_optimizer.param_groups[0]['lr'],
            'fake_images': wandb.Image(fake_images),
        })

        # Save model
        os.makedirs('weights', exist_ok=True)
        torch.save({
            'generator_state_dict': generator.state_dict(),
            'discriminator_state_dict': discriminator.state_dict(),
            'epoch': eph,
            'real_score': real_score,
            'fake_score': fake_score,
        }, os.path.join('weights', 'gan.pth'))

    # Save fake images per epoch to gif
    os.makedirs(os.path.join('results'))
    frames[0].save(os.path.join('results', 'train_gan.gif'), 'GIF',
                   save_all=True, append_images=frames, duration=200, loop=0)


def infer(generator: nn.Module, device: torch.device, num_images=100) -> np.ndarray:
    generator.eval()

    z = torch.randn((num_images, 64), device=device)
    with torch.no_grad():
        fake_images = generator(z)
    fake_images = fake_images.reshape((-1, 1, 28, 28))
    fake_images = F.normalize(fake_images, [-1 / 127 - 1], [1 / 127]).round().to(torch.uint8)
    fake_images = torchvision.utils.make_grid(fake_images, 10, pad_value=110)
    fake_images = np.ascontiguousarray(fake_images.permute((1, 2, 0)).cpu().numpy())
    return fake_images


if __name__ == '__main__':
    config = {
        'batch_size': 256,
        'epoch': 500,
        'lr': 0.0002,
        'reproducibility': True,
        'num_workers': 8,
        'pin_memory': True,
        'prefetch_factor': 60000,
        'persistent_workers': True,
    }
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Model
    generator = nn.Sequential(
        nn.Linear(64, 256),
        nn.ReLU(),
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Linear(256, 784),
        nn.Tanh(),
    ).to(device)
    discriminator = nn.Sequential(
        nn.Linear(784, 256),
        nn.LeakyReLU(0.2),
        nn.Linear(256, 256),
        nn.LeakyReLU(0.2),
        nn.Linear(256, 1),
        nn.Sigmoid(),
    ).to(device)

    if os.path.exists(os.path.join('weights', 'gan.pth')):
        checkpoint = torch.load(os.path.join('weights', 'gan.pth'))
        generator.load_state_dict(checkpoint['generator_state_dict'])

        fake_images = infer(generator, device)

        os.makedirs('results', exist_ok=True)
        cv2.imwrite(os.path.join('results', 'gan.png'), fake_images)
        print('Image creation complete.')
    else:
        # Pytorch reproducibility
        if config['reproducibility']:
            torch.manual_seed(0)
            torch.cuda.manual_seed(0)
            torch.cuda.manual_seed_all(0)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            np.random.seed(0)
            random.seed(0)
            torch.use_deterministic_algorithms(True)
            os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

        # Data
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5,), (0.5,)),
        ])
        trainset = torchvision.datasets.MNIST('data', True, transform, download=True)
        trainloader = torch.utils.data.DataLoader(
            trainset,
            config['batch_size'],
            shuffle=True,
            num_workers=config['num_workers'],
            prefetch_factor=config['prefetch_factor'],
            persistent_workers=config['persistent_workers']
        )

        # Loss function, optimizer
        criterion = nn.BCELoss()
        g_optimizer = torch.optim.RAdam(generator.parameters(), config['lr'])
        d_optimizer = torch.optim.RAdam(discriminator.parameters(), config['lr'])

        # Train
        wandb.init(project='gan', entity='syshin-cubox-ai', config=config)
        train(config['epoch'], trainloader, generator, discriminator, criterion, g_optimizer, d_optimizer, device)
        wandb.finish()
