import os
import cv2
import albumentations as A
from tqdm import tqdm
from config import *


def data_augmentation(image_path, aug_num):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    augmentation = A.Compose([
        A.RandomRotate90(),
        A.Flip(),
        A.Transpose(),
        A.OneOf([
            A.IAAAdditiveGaussianNoise(),
            A.GaussNoise(),
        ], p=0.2),
        A.OneOf([
            A.MotionBlur(p=0.2),
            A.MedianBlur(blur_limit=3, p=0.1),
            A.Blur(blur_limit=3, p=0.1),
        ], p=0.2),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
        A.OneOf([
            A.OpticalDistortion(p=0.3),
            A.GridDistortion(p=0.1),
            A.IAAPiecewiseAffine(p=0.3),
        ], p=0.2),
        A.OneOf([
            A.CLAHE(clip_limit=2),
            A.IAASharpen(),
            A.IAAEmboss(),
            A.RandomBrightnessContrast(),
        ], p=0.3),
        # A.HueSaturationValue(p=0.3),
    ], p=0.5)
    patches = []
    for _ in range(aug_num):
        patches.append(augmentation(image=image)['image'])
    return patches


def create_patches(unlabeled_dir, patches_root, aug_num=12):
    print("doing data augmentation ...")
    # unlabeled_dir = 'aft_data/unlabeled'
    # patches_root = 'aft_data/patches'
    if not os.path.exists(patches_root):
        print("create patches root.")
        # print("create patches root.")
        os.makedirs(patches_root)
    for category_dir in os.listdir(unlabeled_dir):
        for image_dir in tqdm(os.listdir(os.path.join(unlabeled_dir, category_dir))):
            if not os.path.exists(os.path.join(patches_root, image_dir[:-4])):
                os.makedirs(os.path.join(patches_root, image_dir[:-4]))
            image_path = os.path.join(unlabeled_dir, category_dir, image_dir)
            patches = None
            try:
                patches = data_augmentation(image_path, aug_num)
            except:
                print(image_path)
            for i in range(len(patches)):
                patch_name = str(i + 1) + image_dir[-4:]
                patch_dir = os.path.join(patches_root, image_dir[:-4], patch_name)
                cv2.imwrite(patch_dir, patches[i][:, :, [2, 1, 0]])
    print("data augmentation done ... ")


if __name__ == '__main__':
    create_patches(CANDIDATE_ROOT, PATCH_ROOT)
