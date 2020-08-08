# encoding=utf-8
import random
# import visdom
import torch
from torch import nn, optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader, SequentialSampler
from resnet import ResNet18
from lossnet import LossNet
from config import *
from aft import *
from torch import backends
from torch import cuda

iters = 0


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


def loss_pred_loss(inputs, target, margin=1.0, reduction='mean'):
    assert len(inputs) % 2 == 0, 'the batch size is not even.'
    assert inputs.shape == inputs.flip(0).shape
    # flip()翻转
    inputs = (inputs - inputs.flip(0))[
            :len(inputs) // 2]  # [l_1 - l_2B, l_2 - l_2B-1, ... , l_B - l_B+1], where batch_size = 2B
    target = (target - target.flip(0))[:len(target) // 2]
    target = target.detach()

    # 将输入input张量每个元素的夹紧到区间 [min,max][min,max]，并返回结果到一个新张量。
    one = 2 * torch.sign(torch.clamp(target, min=0)) - 1  # 1 operation which is defined by the authors
    loss = None

    if reduction == 'mean':
        loss = torch.sum(torch.clamp(margin - one * inputs, min=0))
        loss = loss / inputs.size(0)  # Note that the size of input is already halved
    elif reduction == 'none':
        loss = torch.clamp(margin - one * inputs, min=0)
    else:
        NotImplementedError()
    return loss


def train_epoch(models, crit, opts, loaders, epoch, epoch_loss, devices, v=None, plot=None):
    global iters
    models['classifier'].train()
    models['module'].train()
    m_classifier_loss = None
    m_module_loss = None

    for data in loaders['train']:
        iters += 1
        inpts = data[0].to(devices)
        labls = data[1].to(devices)

        opts['classifier'].zero_grad()
        opts['module'].zero_grad()

        scores, features = models['classifier'](inpts)
        target_loss = crit(scores, labls)

        # if epoch > epoch_loss:
        # After 120 epochs, stop the gradient from the loss prediction module propagated to the target model.
        # 截断反向传播的梯度流
        features[0] = features[0].detach()
        features[1] = features[1].detach()
        features[2] = features[2].detach()
        features[3] = features[3].detach()
        pred_loss = models['module'](features)
        pred_loss = pred_loss.view(pred_loss.size(0))

        m_classifier_loss = torch.sum(target_loss) / target_loss.size(0)
        m_module_loss = loss_pred_loss(pred_loss, target_loss, margin=MARGIN, reduction='mean')
        m_classifier_loss.backward()
        m_module_loss.backward()
        # loss = m_classifier_loss + WEIGHT * m_module_loss
        # loss.backward()
        opts['classifier'].step()
        opts['module'].step()

        # Visualize
        if (iters % 25 == 1) and (v is not None) and (plot is not None):
            plot['X'].append(iters)
            plot['Y'].append([
                m_classifier_loss.item(),
                m_module_loss.item(),
                # loss.item()
            ])
            v.line(
                X=np.stack([np.array(plot['X'])] * len(plot['legend']), 1),
                Y=np.array(plot['Y']),
                opts={
                    'title': 'Loss over Time',
                    'legend': plot['legend'],
                    'xlabel': 'Iterations',
                    'ylabel': 'Loss',
                    'width': 1200,
                    'height': 390,
                },
                win=1
            )
    print("epoch:%d; classification loss:%.4f; loss prediction loss:%.4f"
          % (epoch, m_classifier_loss.item(), m_module_loss.item()))


def test(models, loaders, devices, mode='val'):
    assert mode == 'val' or mode == 'test'
    models['classifier'].eval()
    models['module'].eval()

    total = 0
    correct = 0
    with torch.no_grad():
        for (inpts, labls) in loaders[mode]:
            inpts = inpts.to(devices)
            labls = labls.to(devices)

            scores, _ = models['classifier'](inpts)
            _, preds = torch.max(scores.data, 1)
            total += labls.size(0)
            correct += (preds == labls).sum().item()

    return 100 * correct / total


def train(models, crit, opts, scheds, loaders, num_epochs, epoch_loss, devices, v=None, data=None):
    print('>> Train a Model.')
    best_acc = 0.
    checkpoint_dir = os.path.join('./hyperkvasir', 'train', 'weights')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    for epoch in tqdm(range(num_epochs)):

        train_epoch(models, crit, opts, loaders, epoch, epoch_loss, devices, v, data)

        scheds['classifier'].step()
        scheds['module'].step()

        # Save a checkpoint
        if False and epoch % 5 == 4:
            print("Test model during training.")
            accs = test(models, loaders, devices, 'test')
            if best_acc < accs:
                best_acc = accs
            print('Val Acc: {:.3f} \t Best Acc: {:.3f}'.format(accs, best_acc))
    print('>> Finished.')


def get_uncertainty(models, patch_path, devices, idx):
    models['classifier'].eval()
    models['module'].eval()
    with torch.no_grad():
        uncertainties = []
        for candidate in tqdm(np.array(os.listdir(patch_path))[idx]):
            patches_uncertainty = torch.tensor([]).cuda()
            for patch in os.listdir(os.path.join(patch_path, candidate)):
                image = Image.open(os.path.join(patch_path, candidate, patch))
                image_tensor = image_transform(image)
                image_tensor.unsqueeze_(0)
                image_tensor = image_tensor.to(devices)
                _, features = models['classifier'](image_tensor)
                pred_loss = models['module'](features)
                pred_loss = pred_loss.view(pred_loss.size(0))
                patches_uncertainty = torch.cat((patches_uncertainty, pred_loss), 0)
            candidate_uncertainty = np.mean(patches_uncertainty.cpu().numpy().squeeze())
            uncertainties.append(candidate_uncertainty)
        return np.array(uncertainties)


def get_diversity(models, patch_root, devices, idx):
    candidates_probs = compute_all_probs(models, patch_root, devices, idx)
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


def get_uncertainty_and_diversity(models, patch_path, devices, idx):
    models['classifier'].eval()
    models['module'].eval()
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
                image_tensor = image_tensor.to(devices)
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


if __name__ == '__main__':
    # 随机数种子，确保实验结果可复现
    torch.manual_seed(SEED)
    cuda.manual_seed(SEED)
    cuda.manual_seed_all(SEED)
    np.random.seed(SEED)  # Numpy module.
    random.seed(SEED)  # Python random module.
    backends.cudnn.benchmark = False
    backends.cudnn.deterministic = True

    uncertainty = None
    diversity = None
    entropy = None
    selected_indices = None

    # 训练过程可视化
    # vis = visdom.Visdom(server='http://192.168.195.55', port=10086)
    vis, plot_data = None, None
    # plot_data = {'X': [], 'Y': [], 'legend': ['Classifier Loss', 'Module Loss', 'Total Loss']}

    indices = list(range(get_sample_num(PATCH_ROOT)))

    # 初始化有标签数据集，随机选择K个样本
    random.shuffle(indices)
    labeled_indices = indices[:K]
    print(">> Initializing labeled dataset and unlabeled data pool.")

    # 初始化无标签数据池
    unlabeled_indices = indices[K:]
    print('remaining unlabeled data: ', len(unlabeled_indices))

    # 函数内部定义变量名可以抽象一点，主函数中的变量名最好便于理解
    # 装载好训练数据
    filenames, labels = load_train_data(CANDIDATE_ROOT, PATCH_ROOT, labeled_indices)
    train_dataset = MyDataset(filenames, labels, transform=image_transform)
    print("Training data : ", len(train_dataset))
    train_loader = DataLoader(train_dataset, batch_size=BATCH, pin_memory=True, shuffle=True)
    # 装载好测试数据
    filenames, labels = load_test_data(TEST_ROOT)
    test_dataset = MyDataset(filenames, labels, transform=image_transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH,
                             sampler=SequentialSampler(range(len(test_dataset))),
                             pin_memory=True)

    dataloaders = {'train': train_loader, 'test': test_loader}

    # 定义分类网络模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classifier_network = ResNet18(num_classes=20)

    # 迁移ImageNet预训练参数
    print(">> Loading pretrained model parameters...")
    classifier_dict = classifier_network.state_dict()
    pretrained_dict = torch.load("resnet18.pth")
    parameter_dict = {k: v for k, v in pretrained_dict.items() if k in classifier_dict}
    classifier_dict.update(parameter_dict)
    classifier_network.load_state_dict(classifier_dict)

    # 定义损失预测模块
    loss_module = LossNet()

    # 将模型转移到训练用的设备上
    classifier_network.to(device)
    loss_module.to(device)

    model = {'classifier': classifier_network, 'module': loss_module}

    print(">> Start active learning!")
    for cycle in range(CYCLES):
        # 先训练（分类网络有ImageNet预训练，但损失预测模块还没有进行学习）

        # 定义分类损失函数
        criterion = nn.CrossEntropyLoss(reduction='none')

        # 定义分类网络优化器
        optim_classifier = optim.SGD(model['classifier'].parameters(),
                                     lr=LR_classifier, momentum=MOMENTUM,
                                     weight_decay=WDECAY)

        # 定义损失预测模块优化器
        optim_module = optim.SGD(model['module'].parameters(),
                                 lr=LR_module, momentum=MOMENTUM,
                                 weight_decay=WDECAY)

        optimizers = {'classifier': optim_classifier, 'module': optim_module}

        # 定义学习率调度器,在MILESTONE后学习率降低到原来的10%
        scheduler_classifier = lr_scheduler.MultiStepLR(optim_classifier, milestones=MILESTONE)
        scheduler_module = lr_scheduler.MultiStepLR(optim_module, milestones=MILESTONE)

        schedulers = {'classifier': scheduler_classifier, 'module': scheduler_module}

        train(model, criterion, optimizers, schedulers, dataloaders, EPOCH, EPOCHL, device, vis, plot_data)
        print("Test model after training.")
        acc = test(model, dataloaders, device, mode='test')
        print('Cycle {}/{} || Label set size {}: Test acc {}'
              .format(cycle + 1, CYCLES, len(labeled_indices), acc))

        # Randomly sample 1000 unlabeled data points
        random.shuffle(unlabeled_indices)
        subset_indices = unlabeled_indices[:SUBSET]

        # 计算样本不确定度
        # print("Computing uncertainty...")
        # uncertainty = get_uncertainty(model, PATCH_ROOT, device, unlabeled_indices)
        #
        # 计算样本差异性
        # print("Computing diversity...")
        # diversity = get_diversity(model, PATCH_ROOT, device, unlabeled_indices)
        if STRATEGY == 'hybrid':
            # 计算不确定性和差异性
            print("Computing uncertainty and diversity...")
            uncertainty, diversity = get_uncertainty_and_diversity(model, PATCH_ROOT, device, subset_indices)
            # 结合两种策略筛选
            arg_u = np.argsort(uncertainty)
            arg_d = np.argsort(diversity)
            assert len(arg_u) == len(arg_d)
            rank = [np.argwhere(arg_u == i)[0][0] +
                    np.argwhere(arg_d == i)[0][0]
                    for i in range(len(arg_u))]
            arg_rank = np.argsort(rank)
            # 选择本轮的样本
            print("Selecting valuable samples...")
            selected_indices = list(np.array(subset_indices)[arg_rank[-K:]])

        elif STRATEGY == 'random':
            print("Randomly select samples...")
            indices = subset_indices
            random.shuffle(indices)
            selected_indices = indices[:K]

        elif STRATEGY == 'diversity':
            print("Computing diversity...")
            diversity = get_diversity(model, PATCH_ROOT, device, subset_indices)
            arg_d = np.argsort(diversity)
            selected_indices = list(np.array(subset_indices)[arg_d[-K:]])

        elif STRATEGY == 'loss':
            uncertainty = get_uncertainty(model, PATCH_ROOT, device, subset_indices)
            arg_u = np.argsort(uncertainty)
            selected_indices = list(np.array(subset_indices)[arg_u[-K:]])

        # 新样本添加至有标签数据集
        labeled_indices.extend(selected_indices)

        # 新样本从无标签数据池中移除
        for i in selected_indices:
            unlabeled_indices.remove(i)
        print('remaining unlabeled data: ', len(unlabeled_indices))

        # 更新训练数据集
        filenames, labels = load_train_data(CANDIDATE_ROOT, PATCH_ROOT, labeled_indices)
        train_dataset = MyDataset(filenames, labels, transform=image_transform)
        print("Training data number: ", len(train_dataset))
        dataloaders['train'] = DataLoader(train_dataset, batch_size=BATCH,
                                          pin_memory=True, shuffle=True)

    torch.save({
        'state_dict_classifier': model['classifier'].state_dict(),
        'state_dict_module': model['module'].state_dict()
    },
        './result/weights/active_resnet18.pth')
