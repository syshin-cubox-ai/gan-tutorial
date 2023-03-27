import os
import random

import numpy as np
import torch
import torch.backends.cudnn
import torch.nn as nn
import torch.utils.data
import torchvision
import torchvision.transforms.functional as F
import tqdm

import wandb


class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Linear(16, 5),
        )
        self.decoder = nn.Sequential(
            nn.Linear(5, 16),
            nn.ReLU(),
            nn.Linear(16, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 784),
            nn.Sigmoid(),
            nn.Unflatten(1, (1, 28, 28))
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        # Encoder
        self.encode1 = self.make_double_conv(1, 32)
        self.encode2 = self.make_double_conv(32, 64)
        self.encode3 = self.make_double_conv(64, 128)
        self.encode4 = self.make_double_conv(128, 256)
        self.max_pool = nn.MaxPool2d(2)

        # Decoder
        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2, output_padding=1)
        self.decode3 = self.make_double_conv(256, 128)

        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decode2 = self.make_double_conv(128, 64)

        self.upconv1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.decode1 = self.make_double_conv(64, 32)

        # Classifier
        self.classifier = nn.Conv2d(32, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def make_double_conv(self, in_channels: int, out_channels: int):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        encode1 = self.encode1(x)
        encode2 = self.encode2(self.max_pool(encode1))
        encode3 = self.encode3(self.max_pool(encode2))
        x = self.encode4(self.max_pool(encode3))

        # Decoder
        x = self.decode3(torch.cat([self.upconv3(x), encode3], dim=1))
        x = self.decode2(torch.cat([self.upconv2(x), encode2], dim=1))
        x = self.decode1(torch.cat([self.upconv1(x), encode1], dim=1))

        # Classifier
        x = self.classifier(x)
        x = self.sigmoid(x)
        return x


def add_noise(img: torch.Tensor, intensity=0.35):
    noise = torch.randn_like(img) * intensity
    noisy_img = torch.clamp(img + noise, 0, 1)
    return noisy_img


def train(model, trainloader, criterion, optimizer, device):
    model.train()

    train_loss = torch.zeros(1, device=device)
    for imgs, _ in trainloader:
        imgs = imgs.to(device)
        noisy_imgs = add_noise(imgs)

        optimizer.zero_grad()
        recovered_imgs = model(noisy_imgs)
        loss = criterion(recovered_imgs, imgs)
        loss.backward()
        optimizer.step()

        train_loss += loss

    train_loss /= len(trainloader)
    return train_loss.item()


def evaluate(model, testloader, criterion, device):
    model.eval()

    test_loss = torch.zeros(1, device=device)
    for imgs, _ in testloader:
        imgs = imgs.to(device)
        noisy_imgs = add_noise(imgs)

        with torch.inference_mode():
            recovered_imgs = model(noisy_imgs)
        test_loss += criterion(recovered_imgs, imgs)

    test_loss /= len(testloader)
    return test_loss.item()


def infer(testloader, model, device):
    model.eval()

    img = iter(testloader).__next__()[0][:6].to(device)
    noisy_img = add_noise(img)
    with torch.inference_mode():
        recovered_img = model(noisy_img)

    result_img = torch.cat([torch.stack(i, 0) for i in zip(img, noisy_img, recovered_img)], 0)
    result_img = F.normalize(result_img, [0], [1 / 255]).round().to(torch.uint8)
    result_img = torchvision.utils.make_grid(result_img, 6, pad_value=110)
    result_img = np.ascontiguousarray(result_img.permute((1, 2, 0)).cpu().numpy())
    return result_img


torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
np.random.seed(0)
random.seed(0)
torch.use_deterministic_algorithms(True)
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

if __name__ == '__main__':
    config = {
        'batch_size': 256,
        'epoch': 55,
        'lr': 0.002,
        'num_workers': 8,
        'pin_memory': True,
        'prefetch_factor': 60000,
        'persistent_workers': True,
    }
    # UNet config
    # config.update({
    #     'lr': 0.001,
    # })

    trainset = torchvision.datasets.FashionMNIST('data', True, torchvision.transforms.ToTensor(), download=True)
    testset = torchvision.datasets.FashionMNIST('data', False, torchvision.transforms.ToTensor(), download=True)
    trainloader = torch.utils.data.DataLoader(
        trainset,
        config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=config['pin_memory'],
        prefetch_factor=config['prefetch_factor'],
        persistent_workers=config['persistent_workers']
    )
    testloader = torch.utils.data.DataLoader(
        testset,
        config['batch_size'],
        num_workers=config['num_workers'],
        pin_memory=config['pin_memory'],
        prefetch_factor=config['prefetch_factor'],
        persistent_workers=config['persistent_workers'],
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Autoencoder().to(device)
    # model = UNet().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.RAdam(model.parameters(), config['lr'])
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, 1, 0, config['epoch'])

    wandb.init(project='Denoising', entity='syshin-cubox-ai', config=config)

    for _ in tqdm.tqdm(range(config['epoch']), 'Train'):
        train_loss = train(model, trainloader, criterion, optimizer, device)
        test_loss = evaluate(model, testloader, criterion, device)

        result_img = infer(testloader, model, device)

        wandb.log({
            'train_loss': train_loss,
            'test_loss': test_loss,
            'lr': optimizer.param_groups[0]['lr'],
            'img (original, noisy, recovered)': wandb.Image(result_img),
        })

        scheduler.step()

    wandb.finish()
