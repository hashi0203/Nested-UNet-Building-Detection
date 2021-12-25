import os
import numpy as np
from torchvision.datasets.vision import VisionDataset
import pathlib
import cv2
import albumentations as albu

from config import img_size

# ======================
# Custom dataset
# ======================

# https://github.com/qubvel/segmentation_models.pytorch

class RS21BD(VisionDataset):
    CLASSES = [
        "nonbuilding",
        "building",
    ]

    def __init__(
        self,
        images_dir,
        masks_dir,
        classes=None,
        augmentation=None,
    ):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.rgb_list = self._get_rgb_list()

        # convert str names to class values on masks
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]

        self.augmentation = augmentation

    def _get_rgb_list(self):
        return list(sorted(pathlib.Path(self.images_dir).glob("*.png")))

    def __len__(self):
        return len(self.rgb_list)

    def __getitem__(self, index):
        rgb_path = self.rgb_list[index]
        rgb_fname = rgb_path.stem
        cls_path = os.path.join(self.masks_dir, rgb_fname + '.png')

        #image = imread(str(rgb_path))
        image = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)

        #mask = imread(str(cls_path))
        mask = cv2.imread(str(cls_path), cv2.IMREAD_UNCHANGED)
        mask[mask < 255] = 0  # non-buildings
        mask[mask == 255] = 1  # buildings
        mask = mask.astype('int64')

        # extract certain classes from mask (e.g. building)
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('int64')

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        return image, mask, str(rgb_path)

# ======================
# Augmentation
# ======================

# https://github.com/albumentations-team/albumentations

def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')

def to_tensor_labels(x, **kwargs):
    return x.transpose(2, 0, 1).astype('int64')

def get_training_augmentation():
    train_transform = [
        albu.HorizontalFlip(p=0.5),
        albu.VerticalFlip(p=0.5),
        albu.RandomRotate90(p=0.5),
        albu.Transpose(p=0.5),
        # albu.Resize(img_size,img_size),
        albu.Lambda(image=to_tensor,mask=to_tensor),
    ]
    return albu.Compose(train_transform)

def get_val_augmentation():
    test_transform = [
        # albu.Resize(img_size,img_size),
        albu.Lambda(image=to_tensor,mask=to_tensor),
    ]
    return albu.Compose(test_transform)
