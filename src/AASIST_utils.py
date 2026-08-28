"""
Main script that trains, validates, and evaluates
various models including AASIST.

AASIST
Copyright (c) 2021-present NAVER Corp.
MIT license
"""
import argparse
import json
import os
import sys
import warnings
from importlib import import_module
from pathlib import Path
from shutil import copy
from typing import Dict, List, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter
from torchcontrib.optim import SWA

import pandas 
from tqdm import tqdm
import numpy as np
warnings.filterwarnings("ignore", category=FutureWarning)


def get_model(model_config: Dict, device: torch.device):
    """Define DNN model architecture"""
    module = import_module(model_config["architecture"])
    _model = getattr(module, "Model")
    model = _model(model_config).to(device)
    nb_params = sum([param.view(-1).size()[0] for param in model.parameters()])
    print("no. model params:{}".format(nb_params))

    return model

def GetAASIST(config,device='cuda'):
    config_path = config["AASIST_config_path"]
    with open(config_path, "r") as f_json:
        aasist_config = json.loads(f_json.read())
        aasist_model_config = aasist_config["model_config"]
    aasist = get_model(aasist_model_config, device)
    aasist.load_state_dict(torch.load(config["AASIST_ckpt_path"], map_location=device),strict=False)
    aasist.eval()
    return aasist


