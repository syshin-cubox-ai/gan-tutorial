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


def add_noise(img: torch.Tensor, intensity=0.3):
    noise = torch.rand_like(img) * intensity
    noisy_img = img + noise
    return noisy_img


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
            nn.Linear(64, 12),
            nn.BatchNorm1d(12),
            nn.ReLU(),
            nn.Linear(12, 3),
        )
        self.decoder = nn.Sequential(
            nn.Linear(3, 12),
            nn.ReLU(),
            nn.Linear(12, 64),
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


def infer(testloader, model, device):
    model.eval()

    img = iter(testloader).__next__()[0].to(device)
    noisy_img = add_noise(img)
    with torch.inference_mode():
        recovered_img = model(noisy_img)

    result_img = torch.cat([torch.stack(i, 0) for i in zip(img, noisy_img, recovered_img)], 0)
    result_img = F.normalize(result_img, [0], [1 / 255]).round().to(torch.uint8)
    result_img = torchvision.utils.make_grid(result_img, 3, pad_value=110)
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
        'epoch': 20,
        'lr': 0.005,
        'num_workers': 8,
        'pin_memory': True,
        'prefetch_factor': 60000,
        'persistent_workers': True,
    }
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    testloader = torch.utils.data.DataLoader(testset, 2)

    model = Autoencoder().to(device)
    optimizer = torch.optim.RAdam(model.parameters(), config['lr'])
    criterion = nn.MSELoss()

    wandb.init(project='Denoising', entity='syshin-cubox-ai', config=config)

    for _ in tqdm.tqdm(range(config['epoch']), 'Train', position=0):
        avg_loss = torch.zeros(1, device=device)
        for imgs, _ in tqdm.tqdm(trainloader, 'Batch', leave=False, position=1):
            imgs = imgs.to(device)
            x = add_noise(imgs)

            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, imgs)
            loss.backward()
            optimizer.step()

            avg_loss += loss

        avg_loss /= len(trainloader)
        result_img = infer(testloader, model, device)

        wandb.log({
            'loss': avg_loss.item(),
            'img (original, noisy, recovered)': wandb.Image(result_img),
        })

    wandb.finish()
