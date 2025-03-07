from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import torchvision as tv
import os
import torch
import random
import numpy as np
from PIL import Image
import json

def read_json(file_name):
    with open(file_name) as handle:
        out = json.load(handle)
    return out


class RandomRotation:
    def __init__(self, angles=[0, 90, 180, 270]):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return TF.rotate(x, angle, expand=True)


def get_transform(MAX_DIM):
    def under_max(image):
        if image.mode != 'RGB':
            image = image.convert("RGB")

        shape = np.array(image.size, dtype=np.float)
        long_dim = max(shape)
        scale = MAX_DIM / long_dim

        new_shape = (shape * scale).astype(int)
        image = image.resize(new_shape)

        return image

    train_transform = tv.transforms.Compose([
        RandomRotation(),
        tv.transforms.Lambda(under_max),
        tv.transforms.ColorJitter(brightness=[0.5, 1.3], contrast=[
            0.8, 1.5], saturation=[0.2, 1.5]),
        tv.transforms.RandomHorizontalFlip(),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    val_transform = tv.transforms.Compose([
        tv.transforms.Lambda(under_max),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    return train_transform, val_transform


transform_class = tv.transforms.Compose([
    tv.transforms.Resize(224),
    tv.transforms.CenterCrop((224, 224)),
    tv.transforms.ToTensor(),
    tv.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


class XrayDataset(Dataset):
    def __init__(self, root, ann, transform_class=transform_class, data_dir=None):
        super().__init__()

        self.root = root
        self.transform_class = transform_class
        self.annot = ann
        self.data_dir = data_dir
      
    def _process(self, image_id):
        val = str(image_id).zfill(12)
        return val + '.jpg'

    def __len__(self):
        return len(self.annot)

    def __getitem__(self, idx):
    
        image_path = self.annot[idx]['image_path']
        image = Image.open(os.path.join(self.data_dir, image_path[0])).resize((300, 300)).convert('RGB')
        image = self.transform_class(image)

        return image

    @staticmethod
    def collate_fn(data):
        class_image_batch = torch.stack(data, 0)
        return class_image_batch


def build_dataset(anno_path=None, data_dir=None):
    all_files = []
    all_files.extend(read_json(anno_path)["train"])
    all_files.extend(read_json(anno_path)["val"])
    all_files.extend(read_json(anno_path)["test"])
    data = XrayDataset(anno_path, all_files,data_dir=data_dir)
    return data


