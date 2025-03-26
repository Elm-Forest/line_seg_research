import albumentations as A
import numpy as np
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def train_augm_d4(sample):
    augms = [
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
    return A.Compose(augms)(image=sample["image"], mask=sample["mask"])


def valid_augm(sample, size=512):
    augms = [A.Resize(height=size, width=size, p=1.0)]
    return A.Compose(augms)(image=sample["image"], mask=sample["mask"])


class PowerLineDataset(Dataset):
    def __init__(self, image_paths, mask_paths, train=True, size=256, num_classes=2):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.num_classes = num_classes
        self.train = train
        self.size = size
        self.resize = transforms.Resize((size, size))
        self.toTensor = transforms.ToTensor()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        mask = Image.open(self.mask_paths[idx]).convert('L')
        img = self.resize(img)
        mask = self.resize(mask)

        img = np.array(img, dtype=np.float32)
        mask = np.array(mask, dtype=np.float32)

        if mask.ndim == 2:
            mask = np.expand_dims(mask, axis=-1)  # (1, H, W)

        if self.num_classes == 2:
            background = 1 - mask
            mask = np.concatenate([background, mask], axis=-1)

        data = {"image": img, "mask": mask}

        if self.train:
            data = train_augm_d4(data)
        else:
            data = valid_augm(data)

        img = TF.to_tensor(data['image'])
        mask = TF.to_tensor(data['mask'])

        return img, mask
