import torch
import torchvision.models as torchmodels
from .caffenet.models import caffenet
from typing import Any, Dict, List, Optional, Tuple


class CaffeNetDiscriminator(torch.nn.Module):
    def __init__(self, num_classes, size=1, depth=1, conv_input=False):

        super().__init__()

        size = int(1024 * size)

        blocks = []
        for d in range(1, depth + 1):
            blocks.append(
                torch.nn.Sequential(
                    torch.nn.Linear(size // d, size // (d + 1)),
                    torch.nn.ReLU(inplace=True), torch.nn.Dropout()))

        if conv_input:
            input_processing = torch.nn.Sequential(
                torch.nn.Flatten(start_dim=1),
                torch.nn.Linear(256 * 6 * 6,
                                4096), torch.nn.ReLU(inplace=True),
                torch.nn.Dropout(), torch.nn.Linear(4096, 4096),
                torch.nn.ReLU(inplace=True), torch.nn.Dropout(),
                torch.nn.Linear(4096, size), torch.nn.ReLU(),
                torch.nn.Dropout())

        else:
            input_processing = torch.nn.Sequential(torch.nn.Linear(4096, size),
                                                   torch.nn.ReLU(),
                                                   torch.nn.Dropout())

        self.layers = torch.nn.Sequential(
            input_processing, *blocks,
            torch.nn.Linear(size // (depth + 1), num_classes))

        # disc head get's default initialization
        print(self.layers)

    def forward(self, x):
        return self.layers(x)


class ResNetDiscriminator(torch.nn.Module):
    def __init__(self, num_classes):

        super().__init__()

        self.layers = torch.nn.Sequential(torch.nn.Linear(512, 1024),
                                          torch.nn.ReLU(), torch.nn.Dropout(),
                                          torch.nn.Linear(1024, 1024),
                                          torch.nn.ReLU(), torch.nn.Dropout(),
                                          torch.nn.Linear(1024, num_classes))

        # disc head get's default initialization

    def forward(self, x):
        return self.layers(x)


class BrainCancerDiscriminator(torch.nn.Sequential):
    """
    Adapted from https://github.com/thuml/Transfer-Learning-Library

    Domain discriminator model from
    `"Domain-Adversarial Training of Neural Networks" <https://arxiv.org/abs/1505.07818>`_
    In the original paper and implementation, we distinguish whether the input features come
    from the source domain or the target domain.

    We extended this to work with multiple domains, which is controlled by the n_domains
    argument.

    Args:
        in_feature (int): dimension of the input feature
        n_domains (int): number of domains to discriminate
        hidden_size (int): dimension of the hidden features
        batch_norm (bool): whether use :class:`~torch.nn.BatchNorm1d`.
            Use :class:`~torch.nn.Dropout` if ``batch_norm`` is False. Default: True.
    Shape:
        - Inputs: (minibatch, `in_feature`)
        - Outputs: :math:`(minibatch, n_domains)`
    """

    def __init__(
        self, in_feature: int, n_domains=70, hidden_size: int = 1024, batch_norm=True, device='cuda:1'
    ):
        self.in_feature = in_feature
        self.n_domains = n_domains
        self.hidden_size = hidden_size
        self.batch_norm = batch_norm
        self.device = device
        super(BrainCancerDiscriminator, self).__init__()
        layers = [
            torch.nn.Linear(in_feature, hidden_size),
            # PrintModule(),
            torch.nn.BatchNorm1d(hidden_size) if self.batch_norm else torch.nn.Dropout(0.5),
            # PrintModule(),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.BatchNorm1d(hidden_size) if self.batch_norm else torch.nn.Dropout(0.5),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(hidden_size, n_domains)  # Output layer for multiple domains
        ]
        self.layers = torch.nn.Sequential(*layers).to(device)
        self.double()
        

    def get_parameters_with_lr(self, lr) -> List[Dict]:
        return [{"params": self.parameters(), "lr": lr}]

    def forward(self, x):
        # x = torch.squeeze(x, dim=0)
        
        if self.training:
            self.train_layer()
        else:
            self.eval_layer()
        # print("HERE==================================", x.shape)
        x = x.view(-1, 64)  # -1 here is for batch size, and 64 is the feature size
        # print("HERE==================================", x.shape, x.dtype)
        x = self.layers(x.double().to(self.device))
        return x
    
    def eval_layer(self):
        self.batch_norm = False
        layers = [
            # PrintModule(),
            torch.nn.Linear(self.in_feature, self.hidden_size),
            # PrintModule(),
            torch.nn.Dropout(0.5),
            # PrintModule(),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(self.hidden_size, self.hidden_size),
            torch.nn.Dropout(0.5),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(self.hidden_size, self.n_domains)  # Output layer for multiple domains
        ]
        self.layers = torch.nn.Sequential(*layers).to(self.device)
        self.double()

    
    def train_layer(self):
        self.batch_norm = True
        layers = [
            torch.nn.Linear(self.in_feature, self.hidden_size),
            # PrintModule(),
            torch.nn.BatchNorm1d(self.hidden_size),
            # PrintModule(),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(self.hidden_size, self.hidden_size),
            torch.nn.BatchNorm1d(self.hidden_size),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(self.hidden_size, self.n_domains)  # Output layer for multiple domains
        ]
        self.layers = torch.nn.Sequential(*layers).to(self.device)
        self.double()


class PrintModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        print("the shape of input is : ", x.shape)
        return x