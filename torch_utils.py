import os

import torch
from torch import nn

def save_model(model: nn.Module, path: str):
    model.cpu()
    _save_model_dir = os.path.dirname(path)
    if(not os.path.exists(_save_model_dir)):
        os.mkdir(_save_model_dir)
    torch.save(model.state_dict(), path)
