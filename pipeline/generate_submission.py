import argparse
import os
from argparse import Namespace

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from pipeline.constants import DEFAULT_DATA_PATH, DEFAULT_EXPERIMENTS_SAVE_PATH
from pipeline.dataset import get_test_loader
from pipeline.models import SignsClassifier
from pipeline.utils import load_json_file


def parse_args() -> Namespace:
    """Сommand line argument parser.

    :return: command line arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='baseline')
    parser.add_argument('--model_name', type=str, default='resnet18')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--device', type=str, default='cuda:0')
    return parser.parse_args()


def get_inference_model(model_name: str, checkpoint_path: str, class2label: dict, device: str) -> torch.nn.Module:
    """Getting model for inference.

    :param model_name: model name
    :param checkpoint_path: path to the model weights
    :param class2label: class to label dict
    :param device: device on which the calculations will be performed
    :return: model
    """
    state_dict = torch.load(checkpoint_path)['state_dict']
    model = SignsClassifier(model_name, len(class2label))
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def get_and_save_test_result(
    exp_name: str,
    model_name: str,
    batch_size: int,
    device: str,
) -> pd.DataFrame:
    """Getting a dataframe with predictions and saving the results to 'submit.csv'.

    :param exp_name: experiment name
    :param model_name: model name
    :param batch_size: batch size
    :param device: device on which the calculations will be performed
    :return: dataframe with predictions
    """
    exp_path = os.path.join(DEFAULT_EXPERIMENTS_SAVE_PATH, exp_name)
    checkpoint_path = os.path.join(exp_path, 'best.pth')

    test_loader = get_test_loader(DEFAULT_DATA_PATH, batch_size, num_workers=6)
    class2label = load_json_file(os.path.join(exp_path, 'class2label.json'))
    label2class = {label: sign_class for sign_class, label in class2label.items()}
    model = get_inference_model(model_name, checkpoint_path, class2label, device)
    pbar = tqdm(test_loader, total=len(test_loader), desc=f'Testing...')
    with torch.no_grad():
        results = []
        for sample in pbar:
            images = sample['image']
            images = images.to(device)
            predictions = model(images)
            predictions = predictions.max(dim=-1)[1].cpu().detach().numpy()
            predictions = np.vectorize(label2class.get)(predictions)
            results.extend(list(zip(sample['fname'], predictions)))
            del predictions, images
    submission_df = pd.DataFrame(results, columns=['filename', 'label'])
    submission_df.to_csv(os.path.join(exp_path, 'submit.csv'), index=False)
    return submission_df


if __name__ == '__main__':
    args = parse_args()
    get_and_save_test_result(args.exp_name, args.model_name, args.batch_size, args.device)
