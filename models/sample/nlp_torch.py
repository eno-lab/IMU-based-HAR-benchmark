import torch
from torch import nn

class SimpleNLP(nn.Module):

    def __init__(self, x_shape, n_classes) -> None:
        super().__init__()
        
        self.nlp1 = nn.Linear(x_shape[1]*x_shape[2], 1024)
        self.nlp2 = nn.Linear(1024, 2048)
        self.nlp3 = nn.Linear(2048, n_classes)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.nlp1(x)
        x = torch.relu(x)
        x = self.nlp2(x)
        x = torch.relu(x)
        x = self.nlp3(x)
        x = torch.relu(x)

        return x


def get_config(dataset, lr_magnif=1):
    return {'learning_rate': 0.001 * lr_magnif, 'regularization_rate': None}


def gen_model(input_shape, n_classes, out_loss, out_activ, metrics, config):
    return SimpleNLP(input_shape, n_classes)


def gen_preconfiged_model(input_shape, n_classes, out_loss, out_activ, dataset, metrics=['accuracy'], lr_magnif=1):
    config = get_config(dataset, lr_magnif)
    return SimpleNLP(input_shape, n_classes), config


def get_optim_config(dataset, trial, lr_magnif=1):
    raise NotImplementedError("No config for optimization")


def get_dnn_framework_name():
    return 'pytorch'
