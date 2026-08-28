import sys
import os
import numpy as np
import torch
from torch import nn
from torch import Tensor
import pandas as pd
from tools.utils import getTrainData,parse_addarguments

# from tools.analysis_level_diff import XLSRLayerExtractor,analyze_xlsr_transition
from tools.analysis_level_diff_attack_wise import XLSRLayerExtractor,run_attackwise_analysis
from core_scripts.startup_config import set_random_seed
from tqdm import tqdm
from data import getDataLoader,evaluate,evaluate_synthesiser
import subprocess
import eval_metrics_DF as em
import fairseq
# from src import AASIST_utils
from src.utils import write_exp_detail
import time

from tools.utils import readconfig
if __name__ == '__main__':
    args = parse_addarguments()
    config_path = args.config
    config = readconfig(config_path)
    #make experiment reproducible
    set_random_seed(config['seed'])

    #define model saving path
    expname = config['exp_name']
    model_save_path = os.path.join('train_res',expname)
    #set model save directory
    if not os.path.exists(model_save_path):
        os.mkdir(model_save_path)
    subprocess.run(['cp',config_path,model_save_path])

    #GPU device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'                  
    print('Device: {}'.format(device))
    # Training and validation 
    num_epochs = config['num_epochs']
    batch_size = config['batch_size']
    # ssl_model = XLSRLayerExtractor(
    # cp_path="/root/rivermind-fs/xlsr2_300m.pt",
    # device=device
    # )
    devloader = getDataLoader(args,
                            datapath=os.path.join(config['dev_data_path']),
                            # datapath=os.path.join(root_path,'data'),
                            labelspath=os.path.join(config['dev_label_path']),
                            batchsize=batch_size*2,
                            shuffle=True,
                            num_workers=8,
                            drop_last=False,
                            stage= "val",
                            )


    # summary_df = analyze_xlsr_transition(
    #     dev_loader=devloader,
    #     ssl_model=ssl_model,
    #     device=device,
    #     save_dir="./xlsr_transition_mlaad_de_pad_repeate",
    # )
    results = run_attackwise_analysis(
        dev_loader=devloader,
        cp_path="/root/rivermind-fs/xlsr2_300m.pt",
        device=device,
        save_dir="./xlsr_transition_decro_en_attackwise_pad_repeate_more",
        save_csv=True,
    )