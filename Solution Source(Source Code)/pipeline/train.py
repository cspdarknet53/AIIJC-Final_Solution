import argparse
import os
import warnings
from argparse import Namespace
from collections import defaultdict
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from pipeline.constants import DEFAULT_EXPERIMENTS_SAVE_PATH, DEFAULT_DATA_PATH, SEED
from pipeline.dataset import get_class2label, get_loaders
from pipeline.models import SignsClassifier
from pipeline.utils import set_global_seed, dump_to_json_file, load_json_file


def parse_args() -> Namespace:
    """Сommand line argument parser.

    :return: command line arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='baseline')
    parser.add_argument('--model_name', type=str, default='resnet18')
    parser.add_argument('--n_epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--device', type=str, default='cuda:0')
    return parser.parse_args()


def train(
    exp_name: str,
    model_name: str,
    n_epochs: int,
    batch_size: int,
    device: str,
    data_path: str = None,
) -> Tuple[Dict[str, List[float]], Dict[str, List[float]]]:
    """Model training.

    Training the model, checking the model on the validation dataset, getting metrics, saving the best model weights.

    :param exp_name: experiment name
    :param model_name: model name. The available names are described in the pipeline.models.ENCODERS
    :param n_epochs: number of epochs
    :param batch_size: batch size
    :param device: device on which the calculations will be performed
    :return: training and validation metrics
    """
    model, class2label, exp_path, criterion, optimizer = train_initialization(exp_name, model_name, device, data_path)
    best_f1 = -np.inf
    ie = 0
    if data_path:
        sdict = torch.load(os.path.join(data_path, 'best.pth'))
        best_f1 = sdict['f1']
        ie = sdict['epoch']
        model.load_state_dict(sdict['state_dict'])
        optimizer.load_state_dict(sdict['optimizer_state_dict'])
    train_metrics, valid_metrics = defaultdict(list), defaultdict(list)
    train_loader, valid_loader = get_loaders(DEFAULT_DATA_PATH, batch_size, class2label, num_workers=6)
    for epoch in range(ie, n_epochs):
        epoch_train_metrics = run_one_epoch(epoch, model, train_loader, optimizer, criterion, device, is_train=True)
        update_metrics_by_epoch_metrics(train_metrics, epoch_train_metrics)
        epoch_valid_metrics = run_one_epoch(epoch, model, valid_loader, optimizer, criterion, device, is_train=False)
        update_metrics_by_epoch_metrics(valid_metrics, epoch_valid_metrics)
        curr_f1 = epoch_valid_metrics['f1']
        if curr_f1 > best_f1:
            best_f1 = curr_f1
            torch.save(
                {
                    'state_dict': model.state_dict(),
                    'epoch': epoch,
                    'f1': best_f1,
                    'optimizer_state_dict': optimizer.state_dict(),
                },
                os.path.join(exp_path, 'best.pth'),
            )
            print('Model saved!')
    return train_metrics, valid_metrics


def train_initialization(
    exp_name: str,
    model_name: str,
    device: str,
    data_path: str = None,
) -> Tuple[nn.Module, Dict[str, int], str, nn.Module, torch.optim.Optimizer]:
    """Initializing the classes involved in the training.

    :param exp_name: experiment name
    :param model_name: model name. The available names are described in the pipeline.models.ENCODERS
    :param device: device on which the calculations will be performed
    :return: model, class to label dict, experiment path, criterion and optimizer
    """
    prepare2train()
    class2label = get_class2label(DEFAULT_DATA_PATH)
    if data_path is None:
       exp_path = get_and_make_train_dir(exp_name, class2label)
    else:
       exp_path = os.path.join(DEFAULT_EXPERIMENTS_SAVE_PATH, exp_name)
       class2label = load_json_file(os.path.join(exp_path, 'class2label.json'))
    model = SignsClassifier(model_name, len(class2label))
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    return model, class2label, exp_path, criterion, optimizer


def prepare2train() -> None:
    """Setting the default values of the system before training."""
    warnings.filterwarnings('ignore')
    cv2.setNumThreads(0)
    cv2.ocl.setUseOpenCL(False)
    set_global_seed(SEED)


def get_and_make_train_dir(exp_name: str, class2label: Dict[str, int]) -> str:
    """Preparing and getting a folder for an experiment.

    :param exp_name: experiment name
    :param class2label: class to label dict
    :return: experiment path
    """
    exp_path = os.path.join(DEFAULT_EXPERIMENTS_SAVE_PATH, exp_name)
    os.makedirs(exp_path, exist_ok=True)
    dump_to_json_file(class2label, os.path.join(exp_path, 'class2label.json'))
    return exp_path


def run_one_epoch(
    epoch: int,
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str,
    is_train: bool = True,
) -> dict:
    """Running a model on a single epoch.

    :param epoch: number of the current epoch
    :param model: model
    :param loader: loader
    :param optimizer: optimizer
    :param criterion: criterion
    :param device: device on which the calculations will be performed
    :param is_train: boolean variable indicating whether it is a training or validation
    :return: metrics on a current epoch
    """
    if is_train:
        model.train()
    else:
        model.eval()
    pbar = tqdm(enumerate(loader), total=len(loader), desc=f'Epoch {epoch} {["evaluate", "train"][is_train]}ing')
    with torch.set_grad_enabled(is_train):
        mean_loss = 0
        gt_labels = []
        pred_labels = []
        for num_batch, sample in pbar:
            images, labels = sample['image'], sample['label']
            images = images.to(device)
            predictions = model(images)
            target = labels.to(device)

            loss = criterion(predictions, target)
            mean_loss += loss.item()
            predictions = predictions.argmax(dim=-1).cpu().detach().numpy()
            labels = labels.numpy()
            pred_labels.extend(list(predictions))
            gt_labels.extend(list(labels))
            if is_train:
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            bar_descr = {'loss': f'{mean_loss / (num_batch + 1):.6f}'}
            pbar.set_postfix(**bar_descr)

            del loss, predictions, images, labels, target
        pbar.close()

    prefix = f'{["val", "train"][is_train]}'
    mean_loss /= len(loader)
    f1 = f1_score(gt_labels, pred_labels, average='micro')
    print(f'Epoch {epoch}: {prefix}_f1={f1:.4f}; {prefix}_loss={mean_loss:.6f}')
    return {'f1': f1, 'loss': mean_loss}


def update_metrics_by_epoch_metrics(metrics: Dict[str, List[float]], epoch_metrics: Dict[str, float]) -> None:
    """Updating metrics.

    :param metrics: metrics on all epochs
    :param epoch_metrics: metrics on a single epoch
    :return: updated metrics on all epochs
    """
    for metric_name, metric_vale in epoch_metrics.items():
        metrics[metric_name].append(metric_vale)


if __name__ == '__main__':
    args = parse_args()
    train(args.exp_name, args.model_name, args.n_epochs, args.batch_size, args.device)
