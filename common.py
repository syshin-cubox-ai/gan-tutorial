import os
import random
from typing import Tuple

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


class BaseGAN:
    def __init__(self, config: dict, generator: nn.Module, discriminator: nn.Module, z_dim: int):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.generator = generator.to(self.device)
        self.discriminator = discriminator.to(self.device)
        self.z_dim = z_dim
        self.trainloader = None
        self.criterion = None
        self.g_optimizer = None
        self.d_optimizer = None
        self.z = None

    def build_dataset(self):
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.5], [0.5]),
        ])
        trainset = torchvision.datasets.MNIST('data', True, transform, download=True)
        self.trainloader = torch.utils.data.DataLoader(
            trainset,
            self.config['batch_size'],
            shuffle=True,
            num_workers=self.config['num_workers'],
            pin_memory=self.config['pin_memory'],
            prefetch_factor=self.config['prefetch_factor'],
            persistent_workers=self.config['persistent_workers']
        )

    def build_criterion(self):
        self.criterion = nn.BCELoss()

    def build_optimizer(self):
        self.g_optimizer = torch.optim.RAdam(self.generator.parameters(), self.config['lr'])
        self.d_optimizer = torch.optim.RAdam(self.discriminator.parameters(), self.config['lr'])

    def train_step(
            self,
            imgs: torch.Tensor,
            labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError('Please implement training batch code.')

    def train(self):
        self.build_dataset()
        self.build_criterion()
        self.build_optimizer()

        if self.config['reproducibility']:
            torch.manual_seed(0)
            torch.cuda.manual_seed(0)
            torch.cuda.manual_seed_all(0)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            np.random.seed(0)
            random.seed(0)
            torch.use_deterministic_algorithms(True)
            os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

        wandb.init(project=f'{self.config["model_name"]}', entity='syshin-cubox-ai', config=self.config)

        d_loss = None
        g_loss = None
        real_score = None
        fake_score = None
        frames = []
        for _ in tqdm.tqdm(range(self.config['epoch']), f'{self.config["model_name"]} Train', position=0):
            for imgs, labels in tqdm.tqdm(self.trainloader, 'Batch', leave=False, position=1):
                d_loss, g_loss, real_score, fake_score = self.train_step(imgs, labels)

            # Get sample images
            sample_img = self.infer(4)
            frames.append(Image.fromarray(sample_img))

            # Log to wandb
            wandb.log({
                'd_loss': d_loss.item(),
                'g_loss': g_loss.item(),
                'D(x)': real_score.mean().item(),
                'D(G(z))': fake_score.mean().item(),
                'sample_image': wandb.Image(sample_img),
            })

        # Save model
        os.makedirs('weights', exist_ok=True)
        torch.save({
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
        }, os.path.join('weights', f'{self.config["model_name"].lower()}.pth'))

        # Save sample images per epoch to gif
        os.makedirs('results', exist_ok=True)
        frames[0].save(os.path.join('results', f'train_{self.config["model_name"].lower()}.gif'), 'GIF',
                       save_all=True, append_images=frames, duration=100, loop=0)

        wandb.finish()

    def infer(self, num_rows: int) -> np.ndarray:
        raise NotImplementedError('Please implement inference code.')

    @staticmethod
    def post_process_sample_img(sample_img: torch.Tensor) -> np.ndarray:
        sample_img = F.normalize(sample_img, [-1 / 127 - 1], [1 / 127]).round().to(torch.uint8)
        sample_img = torchvision.utils.make_grid(sample_img, 10, pad_value=110)
        sample_img = np.ascontiguousarray(sample_img.permute((1, 2, 0)).cpu().numpy())
        return sample_img

    def demo(self, num_rows=10):
        checkpoint = torch.load(os.path.join('weights', f'{self.config["model_name"].lower()}.pth'))
        self.generator.load_state_dict(checkpoint['generator_state_dict'])

        sample_img = self.infer(num_rows)

        os.makedirs('results', exist_ok=True)
        cv2.imwrite(os.path.join('results', f'{self.config["model_name"].lower()}.png'), sample_img)
        print('Image creation complete.')

    def select_training_or_demo(self) -> str:
        if os.path.exists(os.path.join('weights', f'{self.config["model_name"].lower()}.pth')):
            return 'Demo'
        else:
            return 'Train'
