# encoding=utf-8

import random
# import visdom
from torch import nn, optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader, SequentialSampler
from network.resnet import ResNet18
from network.lossnet import LossNet
from aft import *
from torch import backends
from torch import cuda
from utils.utils import *
from dataset import *
from utils.augmentation import create_patches
from active import active_sampling

iters = 0


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


def train(models, crit, opts, scheds, loaders, num_epochs, devices):
    print('>> Train a Model.')
    best_acc = 0.
    checkpoint_dir = os.path.join('./hyperkvasir', 'train', 'weights')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    for epoch in tqdm(range(num_epochs)):

        train_epoch(models, crit, opts, loaders, epoch, devices)

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


def main():
    # Control the random seeds
    torch.manual_seed(SEED)
    cuda.manual_seed(SEED)
    cuda.manual_seed_all(SEED)
    np.random.seed(SEED)  # Numpy module.
    random.seed(SEED)  # Python random module.
    backends.cudnn.benchmark = False
    backends.cudnn.deterministic = True
    print(">> Set random seed: {}".format(SEED))

    # Write data filenames and labels to a txt file.
    read_filenames_and_labels_to_txt(CANDIDATE_ROOT, "gt.txt")

    # Conduct data augmentation first
    create_patches(CANDIDATE_ROOT, PATCH_ROOT)

    # Get the size of the unlabeled data pool to build a list of indices
    indices = list(range(get_sample_num(PATCH_ROOT)))
    # Randomly select K samples in the first cycle
    random.shuffle(indices)
    labeled_indices = indices[:K]
    unlabeled_indices = indices[K:]

    # Load training and testing data
    filenames, labels = load_train_data(CANDIDATE_ROOT, PATCH_ROOT, labeled_indices)
    train_dataset = MyDataset(filenames, labels, transform=image_transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH, shuffle=True, pin_memory=True)
    print("Current training dataset size: {}".format(len(train_dataset)))
    filenames, labels = load_test_data(TEST_ROOT)
    test_dataset = MyDataset(filenames, labels, transform=image_transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH,
                             sampler=SequentialSampler(range(len(test_dataset))),
                             pin_memory=True)
    dataloaders = {'train': train_loader, 'test': test_loader}

    # Set the device for running the network
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build the network structure
    classifier_network = ResNet18(num_classes=23)
    classifier_network.to(device)
    loss_network = LossNet()
    loss_network.to(device)
    # Load pre-trained weight of the classifier network
    classifier_dict = classifier_network.state_dict()
    pretrained_dict = torch.load("resnet18.pth")
    parameter_dict = {k: v for k, v in pretrained_dict.items() if k in classifier_dict}
    classifier_dict.update(parameter_dict)
    classifier_network.load_state_dict(classifier_dict)
    # Integration
    model = {'classifier': classifier_network, 'module': loss_network}

    # Set the loss criterion of the training procedure
    criterion = nn.CrossEntropyLoss(reduction='none')

    print(">> Start active learning!")
    for cycle in range(CYCLES):
        # for each cycle, we need new optimizers and learning rate schedulers
        optim_classifier = optim.SGD(model['classifier'].parameters(),
                                     lr=LR_classifier, momentum=MOMENTUM,
                                     weight_decay=WDECAY)
        optim_loss = optim.SGD(model['module'].parameters(),
                               lr=LR_loss, momentum=MOMENTUM,
                               weight_decay=WDECAY)
        optimizers = {'classifier': optim_classifier, 'loss': optim_loss}
        scheduler_classifier = lr_scheduler.MultiStepLR(optim_classifier, milestones=MILESTONE)
        scheduler_loss = lr_scheduler.MultiStepLR(optim_loss, milestones=MILESTONE)
        schedulers = {'classifier': scheduler_classifier, 'module': scheduler_loss}

        # Training
        train(model, criterion, optimizers, schedulers, dataloaders, EPOCH, device)
        acc = test(model, dataloaders, device, mode='test')
        print('Cycle {}/{} || Label set size {}: Test acc {}'
              .format(cycle + 1, CYCLES, len(labeled_indices), acc))

        # Random subset sampling to explore the data pool
        random.shuffle(unlabeled_indices)
        subset_indices = unlabeled_indices[:SUBSET]

        # Choose the active learning strategy
        selected_indices = active_sampling(strategy="hybrid", model=model, indices=subset_indices)

        # Add new labeled samples to the labeled dataset
        labeled_indices.extend(selected_indices)
        # Remove labeled samples from the unlabeled data pool
        for i in selected_indices:
            unlabeled_indices.remove(i)

        # Update the training dataset
        filenames, labels = load_train_data(CANDIDATE_ROOT, PATCH_ROOT, labeled_indices)
        train_dataset = MyDataset(filenames, labels, transform=image_transform)
        print("Training data number: ", len(train_dataset))
        dataloaders['train'] = DataLoader(train_dataset, batch_size=BATCH,
                                          pin_memory=True, shuffle=True)

        # Save the model of the current cycle
        torch.save(model["classifier"].state_dict(),
                   'checkpoints/active_resnet18_cycle{}.pth'.format(cycle))


if __name__ == '__main__':
    main()
