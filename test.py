import sys
import os
import numpy as np
import torch
from torch import nn
from torch import Tensor
import pandas as pd
from tools.utils import getTrainData,parse_addarguments
from data_utils_SSL import genDataloader
from src.model import Model,SSLModel
from src.utils import write_exp_detail
from tensorboardX import SummaryWriter
from core_scripts.startup_config import set_random_seed
from tqdm import tqdm
from data import getDataLoader,evaluate,evaluate_synthesiser,evaluate_eer
import subprocess
import eval_metrics_DF as em
import fairseq
import time
def evaluate_accuracy(dev_loader, model, device,sslModel):
    val_loss = 0.0
    num_correct = 0.0
    num_total = 0.0
    model.eval()
    weight = torch.FloatTensor([0.1, 0.9]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    bona_data = []
    spoof_data = []
    for batch in tqdm(dev_loader):
        batch_x, batch_y = batch['audio'], batch['label']
        batch_size = batch_x.size(0)
        num_total += batch_size
        batch_x = batch_x.to(device)
        batch_y = batch_y.view(-1).type(torch.int64).to(device)
        batch_x = sslModel(batch_x.squeeze(-1),mask=False, features_only=True)['layer_results']
        batch_out = model(batch_x)
        batch_score = (batch_out[:, 1]
                       ).data.cpu().numpy().ravel() 
        for i in range(batch_size):
            if batch_y[i] == 1:
                bona_data.append(batch_score[i])
            else:
                spoof_data.append(batch_score[i])
        _, batch_pred = batch_out.max(dim=1)

        num_correct += (batch_pred == batch_y).sum(dim=0).item()
        batch_loss = criterion(batch_out, batch_y)
        val_loss += (batch_loss.item() * batch_size)


    val_loss /= num_total
    bona_data = pd.array(bona_data)
    spoof_data = pd.array(spoof_data)
    eer = em.compute_eer(bona_data,spoof_data)[0]
    eer = eer*100
    return val_loss, eer


def evaluate_layers(loader, model, device):
    model.eval()

    bona_data = [[] for _ in range(24)]
    spoof_data = [[] for _ in range(24)]

    for batch in tqdm(loader):
        batch_x = batch["audio"].to(device)
        batch_y = batch["label"].view(-1).long().to(device)

        outputs, _ = model(batch_x)
        labels = batch_y.cpu().numpy()

        for l in range(24):
            scores = outputs[:, l, 1].detach().cpu().numpy()

            bona_data[l].extend(scores[labels == 1])
            spoof_data[l].extend(scores[labels == 0])

    eers = []

    print("\n========== Layer-wise EER ==========")

    for l in range(24):
        bona = np.asarray(bona_data[l])
        spoof = np.asarray(spoof_data[l])

        eer = em.compute_eer(bona, spoof)[0] * 100
        eers.append(eer)

        print(f"L{l + 1:02d}: {eer:.4f}%")

    best_layer = int(np.argmin(eers))

    print("------------------------------------")
    print(f"Best layer: L{best_layer + 1}")
    print(f"Best EER: {eers[best_layer]:.4f}%")

    return eers


from tools.utils import readconfig
if __name__ == '__main__':
    args = parse_addarguments()
    config_path = args.config
    test_scores_root_path = "/root/rivermind-data/SLS/scores"
    config = readconfig(config_path)
    #make experiment reproducible
    set_random_seed(config['seed'])

    #define model saving path
    expname = config['exp_name']
    model_save_path = os.path.join('models',"multilanguage_add_model_baseline","best_model.pth")

    #GPU device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'                  
    print('Device: {}'.format(device))
    
    # nb_params = sum([param.view(-1).size()[0] for param in model.parameters()])
    cp_path = config['xlsr_ckpt_path']
    # sslModel,_,_ = fairseq.checkpoint_utils.load_model_ensemble_and_task([cp_path])
    # sslModel = sslModel[0]
    # sslModel = nn.DataParallel(sslModel).to(device).eval()
    sslModel = None
    batch_size = config['batch_size']

    
    # from src.model_layer_probe import Model 
    model = Model(device, config)

    # model = Model(device,config=config).to(device)
    model = nn.DataParallel(model).to(device)
    checkpoint_path = config['checkpoint_path'] if config["checkpoint_path"].endswith("pth") else os.path.join(config['checkpoint_path'],"best_model.pth")
    model.load_state_dict(torch.load(checkpoint_path),strict=False)
    model.eval()

    write_exp_detail(config,test_scores_root_path)
    
    evalloader = getDataLoader(args,
                        datapath=os.path.join(config['test_data_path']),
                        labelspath=os.path.join(config['test_label_path']),
                        batchsize=32,
                        shuffle=False,
                        num_workers=0,
                        #  dataset_collate_fn=en_dev_collate_fn, # 数据加载特殊处理函数
                        stage= "test",
                        drop_last=False,
                        )
    eers = evaluate_layers(evalloader,model,device)
    print(eers)
    eers = evaluate_eer(evalloader, model, device,sslModel,useeer=True)
    print(eers)
    
    with open(os.path.join(test_scores_root_path, config['exp_name'], 'scores.txt'), 'a') as f:
        labels_path = config["test_label_path"]
        f.write(f"time: {time.ctime()}\n")
        f.write(f"labels_path: {labels_path}\n")
        f.write("EER Results:\n")

        # 获取所有攻击名（可以保留 total 或单独处理，这里全部一起分组）
        attacks = list(eers.keys())
        # 每行最多5个
        chunk_size = 5
        for i in range(0, len(attacks), chunk_size):
            chunk = attacks[i:i+chunk_size]
            line = ",  ".join(f"{name}: {eers[name]:.4f}" for name in chunk)
            f.write(f"  {line}\n")
        f.write("\n")
    print(config["exp_name"])