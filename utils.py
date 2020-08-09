# encoding=utf-8

import torchvision.transforms as t
from PIL import Image
from torch.utils.data import Dataset
import os
from config import CANDIDATE_ROOT, CATEGORY_MAPPING
from tqdm import tqdm


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


def read_filenames_and_labels_to_txt(images_dir, target_txt):
    for category_dir in os.listdir(images_dir):
        for image_dir in tqdm(os.listdir(os.path.join(images_dir, category_dir))):
            with open(target_txt, "w", encoding="utf-8") as f:
                f.write("{},{}".format(image_dir[:-4], CATEGORY_MAPPING[category_dir]))



# def get_not_rgb_samples(path):
#     for category in os.listdir(path):
#         for image in os.listdir(os.path.join(path, category)):



if __name__ == '__main__':
    # print(get_dataset_size(CANDIDATE_ROOT))
    get_bad_samples(CANDIDATE_ROOT)
