import torch
import random
import numpy as np
from torchvision import transforms
from PIL import Image

class Dataset(torch.utils.data.Dataset):
    def __init__(self, root='', batch_size=1, crop_size=0):
        self.root = root
        self.batch_size = batch_size
        self.crop_size = crop_size
        self._init()

    def _init(self):
        self.to_tensor = transforms.ToTensor()

        if self.root.endswith('.csv'):
            import pandas as pd

            # Load CSV into numpy array
            data = pd.read_csv(self.root, header=None).values

            # Normalize to 0–255 if necessary
            if data.max() > 1:
                data = ((data - data.min()) / (data.max() - data.min())) * 255
            data = data.astype(np.uint8)

            # Handle grayscale or RGB input
            if len(data.shape) == 2:
                # Grayscale image: [H, W] → [H, W, 1]
                data = np.expand_dims(data, axis=-1)

            # Ensure it's in [H, W, 3]
            if data.shape[-1] == 1:
                data = np.repeat(data, 3, axis=-1)  # Grayscale to RGB
            elif data.shape[-1] != 3:
                raise ValueError(f"CSV data must have 1 (grayscale) or 3 (RGB) channels. Got: {data.shape[-1]}")

            # Convert to tensor [C, H, W], then [1, C, H, W]
            image_tensor = torch.tensor(data, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
            image_tensor = (image_tensor / 127.5) - 1.0  # Normalize to [-1, 1]
            self.image = image_tensor

        else:
            # For image files
            image = Image.open(self.root).convert('RGB')
            self.image = self.to_tensor(image).unsqueeze(0)
            self.image = (self.image - 0.5) * 2

        self.reals = None
        self.noises = None
        self.amps = None

    def _get_augment_params(self, size):
        random.seed(random.randint(0, 12345))
        w_size, h_size = size
        x = random.randint(0, max(0, w_size - self.crop_size))
        y = random.randint(0, max(0, h_size - self.crop_size))
        flip = random.random() > 0.5
        return {'pos': (x, y), 'flip': flip}

    def _augment(self, image, aug_params, scale=1):
        x, y = aug_params['pos']
        image = image[:, round(x * scale):(round(x * scale) + round(self.crop_size * scale)),
                      round(y * scale):(round(y * scale) + round(self.crop_size * scale))]
        if aug_params['flip']:
            image = image.flip(-1)
        return image

    def __getitem__(self, index):
        amps = self.amps
        if self.crop_size:
            reals, noises = {}, {}
            aug_params = self._get_augment_params(self.image.size()[-2:])
            for key in self.reals.keys():
                scale = self.reals[key].size(-1) / float(self.image.size(-1))
                reals[key] = self._augment(self.reals[key].clone(), aug_params, scale)
                noises[key] = self._augment(self.noises[key].clone(), aug_params, scale)
        else:
            reals = self.reals
            noises = self.noises

        return {'reals': reals, 'noises': noises, 'amps': amps}

    def __len__(self):
        return self.batch_size
