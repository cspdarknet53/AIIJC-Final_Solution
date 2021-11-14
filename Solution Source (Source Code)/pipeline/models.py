from functools import partial

import torch
from torch import nn
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.nn.modules.pooling import AdaptiveAvgPool2d
from efficientnet_pytorch import EfficientNet

ENCODERS = {
    
    'efficientnet': {
        'features': 704,
        'init_op': EfficientNet.from_pretrained('efficientnet-b2',
    },
    'efficientnetb7': {
        'features': 1408,
        'init_op': EfficientNet.from_pretrained('efficientnet-b7'),
    },
    'efficientnetl2': {
        'features': 1408,
        'init_op' : EfficientNet.from_name('efficientnet-l2')
    }
}


class SignsClassifier(nn.Module):
    """
    A model for classifying signs.
    """

    def __init__(self, encoder_name: str, n_classes: int, dropout_rate: float = 0.0):
        """Initializing the class.

        :param encoder_name: name of the network encoders
        :param n_classes: number of output classes
        :param dropout_rate: dropout rate
        """
        super().__init__()
        self.encoder = ENCODERS[encoder_name]['init_op']
        self.avg_pool = AdaptiveAvgPool2d((1, 1))
        self.dropout = Dropout(dropout_rate)
        self.cv1 = nn.Conv2D(1408,704,3)
        self.cv2 = nn.Conv2D(1408,704,3)
        self.cv3 = nn.Conv2D(1408,704,3)
        self.cv4 = nn.Conv2D(1408,704,3)
        self.cv5 = nn.Conv2D(1408,704,3)
        self.fc = Linear(ENCODERS[encoder_name]['features'], 5)
        self.fc1 = Linear(ENCODERS[encoder_name]['features'], 6)
        self.fc2 = Linear(ENCODERS[encoder_name]['features'], 6)
        self.fc3 = Linear(ENCODERS[encoder_name]['features'], 6)
        self.fc4 = Linear(ENCODERS[encoder_name]['features'], 6)
        self.fc5 = Linear(ENCODERS[encoder_name]['features'], 6)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Getting the model prediction.

        :param x: input batch tensor
        :return: prediction
        """
        x = self.encoder(inputs=x)
        x1 = self.avg_pool(x)
        x1 = nn.Softmax(self.fc(x1),dim=-1)

        x2 = nn.ReLU(self.cv1(x))
        x2 = self.avg_pool(x2)
        x2 = nn.Sigmoid(self.fc1(x2),dim=-1)

        x3 = nn.ReLU(self.cv2(x))
        x3 = self.avg_pool(x3)
        x3 = nn.Sigmoid(self.fc2(x3),dim=-1)

        x4 = nn.ReLU(self.cv3(x))
        x4 = self.avg_pool(x4)
        x4 = nn.Sigmoid(self.fc3(x4),dim=-1)

        x5 = nn.ReLU(self.cv4(x))
        x5 = self.avg_pool(x5)
        x5 = nn.Sigmoid(self.fc4(x5),dim=-1)

        x6 = nn.ReLU(self.cv5(x))
        x6 = self.avg_pool(x6)
        x6 = nn.Sigmoid(self.fc5(x6),dim=-1) 

        return x1,x2,x3,x4,x5,x6
