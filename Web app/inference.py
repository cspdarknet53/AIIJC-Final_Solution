from typing import Callable, Tuple,  Any
from functools import partial
import torch
import json
import numpy as np
import albumentations as albu
from torch import nn
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.nn.modules.pooling import AdaptiveAvgPool2d
from timm.models.resnet import resnet18, resnet34, resnet50
from timm.models.efficientnet import efficientnet_b2

ENCODERS = {                                                    # Encoding For Different Models
    'resnet18': {
        'features': 512,
        'init_op': partial(resnet18, pretrained=True),
    },
    'resnet34': {
        'features': 512,
        'init_op': partial(resnet34, pretrained=True),
    },
    'resnet50': {
        'features': 2048,
        'init_op': partial(resnet50, pretrained=True),
    },
    'efficientnet': {
        'features': 1408,
        'init_op': partial(efficientnet_b2, pretrained=True),
    },
}


class SignsClassifier(nn.Module):
    """
    A model for classifying signs.
    """

    def __init__(self, encoder_name: str, n_classes: int, dropout_rate: float = 0.2):
        """Initializing the class.

        :param encoder_name: name of the network encoder
        :param n_classes: number of output classes
        :param dropout_rate: dropout rate
        """
        super().__init__()
        self.encoder = ENCODERS[encoder_name]['init_op']()
        self.avg_pool = AdaptiveAvgPool2d((1, 1))
        self.dropout = Dropout(dropout_rate)
        self.fc = Linear(ENCODERS[encoder_name]['features'], n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Getting the model prediction.

        :param x: input batch tensor
        :return: prediction
        """
        x = self.encoder.forward_features(x)
        x = self.avg_pool(x).flatten(1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


def load_json_file(path: str) -> Any: # Loading JSON File
    with open(path, 'r') as f:
        data = json.load(f)
    return data

def prep(img, img_size: Tuple[int, int] = (224, 224)) -> Callable: # Preparing Image To Enter The Model
    valid_transform = [
        albu.Resize(img_size[0], img_size[1]),
    ]
    img = albu.Compose(valid_transform)(image=img)['image']
    img = img.astype(np.float32)
    img /= 255
    img = np.transpose(img, (2, 0, 1))
    img -= np.array([0.485, 0.456, 0.406])[:, None, None]
    img /= np.array([0.229, 0.224, 0.225])[:, None, None]

    return img




