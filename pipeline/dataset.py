import os
from typing import Callable, Dict, Mapping, Tuple, Optional, Union

import torch
import albumentations as albu
import cv2
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader

from pipeline.constants import IMG_SIZE, SEED
from pipeline.utils import split_data


def get_class2label(data_path: str) -> Dict[str, int]:
    """Getting a dictionary class to label.

    :param data_path: path to the data folder
    :return: class to label dict
    """
    df = pd.read_csv(os.path.join(data_path, 'train.csv'))
    class2label = {sign_class: label for label, sign_class in enumerate(df['label'].unique())}
    return class2label


def get_loaders(
    data_path: str,
    batch_size: int,
    class2label: Dict[str, int],
    num_workers: int = 6,
) -> Tuple[DataLoader, DataLoader]:
    """Getting training and validation loaders.

    :param data_path: path to the data folder
    :param batch_size: batch size
    :param class2label: class to label dict
    :param num_workers: number of workers
    :return: training and validation loaders
    """
    df = pd.read_csv(os.path.join(data_path, 'train.csv'))
    train_df, valid_df = split_data(df, 'label', SEED, 0.1)
    train_dataset = SignDataset(train_df, data_path, class2label, training_augmentation())
    valid_dataset = SignDataset(valid_df, data_path, class2label, validation_augmentation())
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, valid_loader


def get_test_loader(data_path: str, batch_size: int, num_workers: int = 6) -> DataLoader:
    """Getting test loader.

    :param data_path: path to the data folder
    :param batch_size: batch size
    :param num_workers: number of workers
    :return: test loader
    """
    test_df = pd.read_csv(os.path.join(data_path, 'sample_submission.csv'))
    test_dataset = SignTestDataset(test_df, data_path)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return test_loader


def read_rgb_img(img_path: str) -> np.ndarray:
    """Getting image in rgb format from a path.

    :param img_path: path to the image
    :return: image in rgb format
    """
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def preprocess_img(img: np.ndarray) -> np.ndarray:
    """Preprocessing an image before training.

    :param img: image before preprocessing
    :return: image after preprocessing
    """
    img = img.astype(np.float32)
    img /= 255
    img = np.transpose(img, (2, 0, 1))
    img -= np.array([0.485, 0.456, 0.406])[:, None, None]
    img /= np.array([0.229, 0.224, 0.225])[:, None, None]
    return img


def validation_augmentation(img_size: Tuple[int, int] = IMG_SIZE) -> Callable:
    """Getting validation/testing augmentation.

    :param img_size: image size after augmentation
    :return: validation/testing augmentation
    """
    valid_transform = [
        albu.Resize(img_size[0], img_size[1]),
    ]
    return albu.Compose(valid_transform)


def training_augmentation(img_size: Tuple[int, int] = IMG_SIZE) -> Callable:
    """Getting training augmentation.

    :param img_size: image size after augmentation
    :return: training augmentation
    """
    train_transform = [
        albu.ImageCompression(quality_lower=60, quality_upper=100, always_apply=True),
        albu.augmentations.transforms.OpticalDistortion(),
        albu.augmentations.transforms.ColorJitter(always_apply=True),
        albu.augmentations.geometric.rotate.Rotate (limit=5),
        albu.augmentations.transforms.RandomShadow (),
        albu.augmentations.transforms.Blur(),
        albu.Resize(img_size[0], img_size[1]),
    ]
    return albu.Compose(train_transform)


class SignDataset(Dataset):
    """
    Dataset for training and validation.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        data_path: str,
        class2label: Mapping[str, int],
        transform: Optional[Callable] = None,
    ):
        """Initializing the class.

        :param df: dataframe with columns 'filename' and 'label'
        :param data_path: path to the data folder
        :param class2label: class to label dict
        :param transform: augmentations to be applied to images in the dataset
        """
        self.df = df.to_records(index=False)
        self.data_path = data_path
        self.class2label = class2label
        self.transform = transform

    def __getitem__(self, index: int) -> Dict[str, Union[str, np.ndarray, int]]:
        data_sample = self.df[index]
        fname, sign_class = data_sample['filename'], data_sample['label']
        if "drive" not in fname:
         img = read_rgb_img(os.path.join(self.data_path, fname))
        else:
         img = read_rgb_img(fname)
        if self.transform is not None:
            img = self.transform(image=img)['image']
        img = preprocess_img(img)
        lbl = self.class2label[sign_class]
        #label = torch.zeros([1,len(self.class2label)]).long()
        #label[:,lbl] = 1
        return {'image': img, 'label': lbl, 'sign_class': sign_class, 'fname': fname}

    def __len__(self) -> int:
        return len(self.df)


class SignTestDataset(Dataset):
    """
    Dataset for testing.
    """

    def __init__(self, df: pd.DataFrame, data_path: str):
        """Initializing the class.

        :param df: dataframe with column 'filename'
        :param data_path: path to the data folder
        """
        self.df = df.to_records(index=False)
        self.data_path = data_path

    def __getitem__(self, index: int) -> Dict[str, Union[str, np.ndarray]]:
        fname = self.df[index]['filename']
        if "drive" not in fname:
         img = read_rgb_img(os.path.join(self.data_path, fname))
        else:
         img = read_rgb_img(fname)
        img = validation_augmentation()(image=img)['image']
        return {'image': preprocess_img(img), 'fname': fname}

    def __len__(self) -> int:
        return len(self.df)
