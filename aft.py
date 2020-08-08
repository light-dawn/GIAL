import torch.nn.functional as F
import numpy as np
from utils import *
from tqdm import tqdm
import torch


def compute_all_probs(model, patch_path, device, idx):
    model['classifier'].eval()
    model['module'].eval()
    with torch.no_grad():
        all_probs = []
        for candidate in tqdm(np.array(os.listdir(patch_path))[idx]):
            candidate_probs = []
            for patch in os.listdir(os.path.join(patch_path, candidate)):
                image = Image.open(os.path.join(patch_path, candidate, patch))
                image_tensor = image_transform(image)
                image_tensor.unsqueeze_(0)
                image_tensor = image_tensor.to(device)
                # 这是特别的模型，第二个输出是特征图
                output_tensor = model['classifier'](image_tensor)[0]
                prob = F.softmax(output_tensor, dim=1)
                candidate_probs.append(prob.cpu().numpy().squeeze())
            all_probs.append(np.array(candidate_probs))
    return np.array(all_probs)


def find_dominant_class(ps):
    probs_array = np.array(ps)
    prob_sum = np.sum(probs_array, axis=0)
    index = np.argmax(prob_sum)
    # 返回最大概率类别的索引
    return index


def sort_probs(ps, di):
    probs_array = np.array(ps)
    dominant_probs = probs_array[:, di]
    sorted_index = np.argsort(-dominant_probs)
    sorted_probs_array = probs_array[list(sorted_index)]
    return sorted_probs_array


def majority(input_p, alpha=0.25):
    top = round(len(input_p) * alpha)
    return input_p[0:top]


# lambda_1:entropy lambda_2:diversity
def compute_diversity(ps, lambda_1=0, lambda_2=1):
    r = np.zeros((len(ps), len(ps)))
    for i in range(0, len(ps)):
        for j in range(0, i + 1):
            temp = 0.0
            if i == j:
                for pi in ps[i]:
                    temp += pi * np.log(pi)
                r[i, i] = - lambda_1 * temp
            else:
                for k in range(len(ps[0])):
                    temp += (ps[i][k] - ps[j][k]) * np.log(ps[i][k] / ps[j][k])
                r[i, j] = lambda_2 * temp
                r[j, i] = r[i, j]
    values = sum(r)
    score = sum(values)
    return score
