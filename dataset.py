# encoding=utf-8

import torchvision.transforms as t
from PIL import Image
from torch.utils.data import Dataset
from config import *
import os
import numpy as np


class MyDataset(Dataset):
    def __init__(self, f_names, ls, transform=None, target_transform=None):
        self.filenames = f_names
        self.transform = transform
        self.target_transform = target_transform
        self.labels = ls
        self.loader = self.image_loader

    def __getitem__(self, item):
        fn = self.filenames[item]
        label = self.labels[item]
        img = self.loader(fn)
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.filenames)

    @staticmethod
    def image_loader(path):
        image = Image.open(path)
        image = image.convert("RGB")
        return image


def image_transform(image):
    transforms = t.Compose(
        [
            t.Resize((224, 224)),
            t.ToTensor()
        ]
    )
    image_tensor = transforms(image)
    return image_tensor


def load_train_data(candidate_path, patch_path, idx):
    candidates = np.array(os.listdir(patch_path))[idx]
    patch_num = len(os.listdir(os.path.join(patch_path, candidates[0])))
    fn = []
    lb = []
    for c in candidates:
        for patch in os.listdir(os.path.join(patch_path, c)):
            fn.append(os.path.join(patch_path, c, patch))
        flag = False
        c_name = c + '.jpg'
        for category in os.listdir(candidate_path):
            for file in os.listdir(os.path.join(candidate_path, category)):
                if c_name == file:
                    lb.extend([CATEGORY_MAPPING[category]]*patch_num)
                    flag = True
                    break
            if flag:
                break
    assert len(fn) == len(lb)
    return fn, lb


def load_test_data(test_path):
    fn = []
    lb = []
    for category in os.listdir(test_path):
        filename_list = os.listdir(os.path.join(test_path, category))
        lb.extend([CATEGORY_MAPPING[category]]*len(filename_list))
        for file in filename_list:
            fn.append(os.path.join(test_path, category, file))
    assert len(fn) == len(lb)
    return fn, lb

