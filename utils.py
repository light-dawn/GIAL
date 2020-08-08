import torchvision.transforms as t
from PIL import Image
from torch.utils.data import Dataset
import os
from config import CANDIDATE_ROOT


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


def get_sample_num(path):
    return len(os.listdir(path))


def get_dataset_size(path):
    size = 0
    for category in os.listdir(path):
        size += len(os.listdir(os.path.join(path, category)))
    return size


def get_bad_samples(path):
    for category in os.listdir(path):
        for image in os.listdir(os.path.join(path, category)):
            if image[-7:-4] == 'gif':
                print(os.path.join(os.path.join(path, category, image)))


# def get_not_rgb_samples(path):
#     for category in os.listdir(path):
#         for image in os.listdir(os.path.join(path, category)):



if __name__ == '__main__':
    # print(get_dataset_size(CANDIDATE_ROOT))
    get_bad_samples(CANDIDATE_ROOT)
