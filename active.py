# encoding=utf-8
import torch
from tqdm import tqdm
import os
from dataset import image_transform
import numpy as np
from PIL import Image
from aft import *
from config import *
import random


def get_uncertainty(models, patch_path, idx):
    models['classifier'].eval()
    models['module'].eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        uncertainties = []
        for candidate in tqdm(np.array(os.listdir(patch_path))[idx]):
            patches_uncertainty = torch.tensor([]).cuda()
            for patch in os.listdir(os.path.join(patch_path, candidate)):
                image = Image.open(os.path.join(patch_path, candidate, patch))
                image_tensor = image_transform(image)
                image_tensor.unsqueeze_(0)
                image_tensor = image_tensor.to(device)
                _, features = models['classifier'](image_tensor)
                pred_loss = models['module'](features)
                pred_loss = pred_loss.view(pred_loss.size(0))
                patches_uncertainty = torch.cat((patches_uncertainty, pred_loss), 0)
            candidate_uncertainty = np.mean(patches_uncertainty.cpu().numpy().squeeze())
            uncertainties.append(candidate_uncertainty)
        return np.array(uncertainties)


def get_diversity(models, patch_root, idx):
    candidates_probs = compute_all_probs(models, patch_root, idx)
    diversities = []
    for probs in candidates_probs:
        # 每个candidate会有一个probs数组，第i行第j个元素代表第i个patch在第j个类别上的预测概率
        # 计算主导类的索引
        dominant_index = find_dominant_class(probs)
        # 按照主导类预测概率大小对probs数组进行排序
        sorted_probs = sort_probs(probs, dominant_index)
        # 传入排好序的probs数组，按照majority selection原则选取其中的一部分
        p = majority(sorted_probs)
        # 将选择的部分patch的概率数组传入，计算这个candidate的样本价值
        diversities.append(compute_diversity(p))
    return np.array(diversities)


def get_uncertainty_and_diversity(models, patch_path, idx):
    models['classifier'].eval()
    models['module'].eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    uncertainties = []
    diversities = []
    all_probs = []
    with torch.no_grad():
        for candidate in tqdm(np.array(os.listdir(patch_path))[idx]):
            candidate_probs = []
            patches_uncertainty = torch.tensor([]).cuda()
            for patch in os.listdir(os.path.join(patch_path, candidate)):
                image = Image.open(os.path.join(patch_path, candidate, patch))
                image_tensor = image_transform(image)
                image_tensor.unsqueeze_(0)
                image_tensor = image_tensor.to(device)
                output_tensor, features = models['classifier'](image_tensor)
                prob = F.softmax(output_tensor, dim=1)
                pred_loss = models['module'](features)
                pred_loss = pred_loss.view(pred_loss.size(0))
                patches_uncertainty = torch.cat((patches_uncertainty, pred_loss), 0)
                candidate_probs.append(prob.cpu().numpy().squeeze())
            all_probs.append(np.array(candidate_probs))
            candidate_uncertainty = np.mean(patches_uncertainty.cpu().numpy().squeeze())
            uncertainties.append(candidate_uncertainty)
    for probs in all_probs:
        # 每个candidate会有一个probs数组，第i行第j个元素代表第i个patch在第j个类别上的预测概率
        # 计算主导类的索引
        dominant_index = find_dominant_class(probs)
        # 按照主导类预测概率大小对probs数组进行排序
        sorted_probs = sort_probs(probs, dominant_index)
        # 传入排好序的probs数组，按照majority selection原则选取其中的一部分
        p = majority(sorted_probs)
        # 将选择的部分patch的概率数组传入，计算这个candidate的样本价值
        diversities.append(compute_diversity(p))
    return uncertainties, diversities


def active_sampling(strategy, indices, model):
    if strategy == 'hybrid':
        print("Computing uncertainty and diversity...")
        uncertainty, diversity = get_uncertainty_and_diversity(model, PATCH_ROOT, indices)
        # 结合两种策略筛选
        arg_u = np.argsort(uncertainty)
        arg_d = np.argsort(diversity)
        assert len(arg_u) == len(arg_d)
        rank = [np.argwhere(arg_u == i)[0][0] +
                np.argwhere(arg_d == i)[0][0]
                for i in range(len(arg_u))]
        arg_rank = np.argsort(rank)
        # 选择本轮的样本
        selected_indices = list(np.array(indices)[arg_rank[-K:]])

    elif STRATEGY == 'random':
        print("Randomly select samples...")
        indices = indices
        random.shuffle(indices)
        selected_indices = indices[:K]

    elif STRATEGY == 'diversity':
        print("Computing diversity...")
        diversity = get_diversity(model, PATCH_ROOT, indices)
        arg_d = np.argsort(diversity)
        selected_indices = list(np.array(indices)[arg_d[-K:]])

    elif STRATEGY == 'loss':
        uncertainty = get_uncertainty(model, PATCH_ROOT, indices)
        arg_u = np.argsort(uncertainty)
        selected_indices = list(np.array(indices)[arg_u[-K:]])

    else:
        print("The strategy is not supported. ")
        selected_indices = None

    return selected_indices
