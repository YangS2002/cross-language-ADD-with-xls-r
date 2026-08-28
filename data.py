import torch
import numpy as np
import random
import os
from torch.utils.data import Dataset
import librosa
from torch import Tensor
from tqdm import tqdm
import pandas
from data_utils_SSL import process_Rawboost_feature
def pad(x, max_len=64600):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    # need to pad
    num_repeats = int(max_len / x_len)+1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    return padded_x	
def getEnData(labels_path,language=''):
    # 生成DECRO的训练数据labels
    # 返回文件名数组和文件名-标签字典
    file_names = []
    filenames_labels = {}
    i = 0
    with open(os.path.join(labels_path),'r') as f:
        file = f.readlines()
    for line in file:
        line = line.split(' ')
        if len(line)<2:
            continue
        # 4 4-727-124443-0055 - baidu spoof 
        # label 文件的结构，每行
        fn = line[1]
        al = line[-2]
        label = 0 if "spoof" in line[-1] else 1
        if language == 'ch':
            if al not in['vits','nvcnet'] :
                continue
        if language == 'ch-bona':
            if al not in['vits','nvcnet'] :
                if label == 0:
                    continue

        filenames_labels[fn] = int(label)
        file_names.append(fn)
    return filenames_labels,file_names


def set_random_seed(random_seed, args=None):
    """ set_random_seed(random_seed, args=None)
    
    Set the random_seed for numpy, python, and cudnn
    
    input
    -----
      random_seed: integer random seed
      args: argue parser
    """
    
    # initialization                                       
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)
    os.environ['PYTHONHASHSEED'] = str(random_seed)

    #For torch.backends.cudnn.deterministic
    #Note: this default configuration may result in RuntimeError
    #see https://pytorch.org/docs/stable/notes/randomness.html    
    if args is None:
        cudnn_deterministic = True
        cudnn_benchmark = False
    else:
        cudnn_deterministic = args.cudnn_deterministic_toggle
        cudnn_benchmark = args.cudnn_benchmark_toggle
    
        if not cudnn_deterministic:
            print("cudnn_deterministic set to False")
        if cudnn_benchmark:
            print("cudnn_benchmark set to True")
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)
        torch.backends.cudnn.deterministic = cudnn_deterministic
        torch.backends.cudnn.benchmark = cudnn_benchmark
    return



def compute_det_curve(target_scores, nontarget_scores):

    n_scores = target_scores.size + nontarget_scores.size
    all_scores = np.concatenate((target_scores, nontarget_scores))
    labels = np.concatenate((np.ones(target_scores.size), np.zeros(nontarget_scores.size)))

    # Sort labels based on scores
    indices = np.argsort(all_scores, kind='mergesort')
    labels = labels[indices]


    # Compute false rejection and false acceptance rates
    tar_trial_sums = np.cumsum(labels)
    nontarget_trial_sums = nontarget_scores.size - (np.arange(1, n_scores + 1) - tar_trial_sums)
    if target_scores.size == 0:
        return 1
    else:
        frr = np.concatenate((np.atleast_1d(0), tar_trial_sums / target_scores.size))  # false rejection rates
    far = np.concatenate((np.atleast_1d(1), nontarget_trial_sums / nontarget_scores.size))  # false acceptance rates
    thresholds = np.concatenate((np.atleast_1d(all_scores[indices[0]] - 0.001), all_scores[indices]))  # Thresholds are the sorted scores

    return frr, far, thresholds

def compute_eer(target_scores, nontarget_scores):
    """ Returns equal error rate (EER) and the corresponding threshold. """
    frr, far, thresholds = compute_det_curve(target_scores, nontarget_scores)
    abs_diffs = np.abs(frr - far)
    min_index = np.argmin(abs_diffs)
    eer = np.mean((frr[min_index], far[min_index]))
    return eer, thresholds[min_index]
from torch.utils.data import Dataset,DataLoader
import torch
import torchaudio
import os
import numpy as np
import random

def getlabels(path,collate_fn=None):
    labels = {}
    labels['labels'] = []
    labels['fns'] = []
    labels['synthesizers'] = []
    labels['language'] = []
    labels['spk_ids'] = []
    spks = {}
    synthesizers = {}
    n = 1 # bona synthesier is 0
    with open(path,'r') as f:
        lines = f.read().split('\n')
        for line in lines:
            line = line.split(' ')
            if len(line) <5:
                continue
            fn = line[1]
            label = line[-1]
            synthesizer = line[3] #合成方法标签
            language = line[2]
            spk = line[0]
            if collate_fn is not None and callable(collate_fn):
                if collate_fn(fn=fn,label=label,synthesizer=synthesizer,language=language):
                    continue 
            labels['fns'].append(fn)
            if label == 'spoof':
                if synthesizer not in synthesizers:
                    synthesizers[synthesizer] = n
                    n += 1
            else: # bona
                synthesizer = 'bonafide'
                if synthesizer not in synthesizers:
                    synthesizers[synthesizer] = 0
            if spk not in spks:
                spks[spk] = len(spks)
            labels['synthesizers'].append(synthesizer)
            labels['language'].append(language)
            labels['spk_ids'].append(spks[spk])
    return labels,n,synthesizers

def getRandomSpeedFactor(min_speed=0.5,max_speed = 2.0):
    randomnum = random.uniform(0.5, 2.0)
    randomspeed = round(randomnum,1)
    return randomspeed


def pad(waveform, max_len=64600):
    channels, time = waveform.shape

    if time < max_len:
        repeat_num = (max_len + time - 1) // time
        waveform = waveform.repeat(1, repeat_num)
        waveform = waveform[:, :max_len]
    else:
        waveform = waveform[:, :max_len]

    return waveform.type(torch.float)

def simple_collate_fn(batch):
    audio = torch.stack([item["audio"] for item in batch])
    label = torch.tensor([item["label"] for item in batch])
    vocoder_label = torch.tensor([item["vocoder_label"] for item in batch])
    file_name = [item['file_name'] for item in batch]
    language = torch.tensor([item['language'] for item in batch])
    spk_id = torch.tensor([item['spk_id'] for item in batch])
    vocoder_n = [item["vocoder_n"] for item in batch]
    return {
        "audio": audio,
        "label": label,
        "vocoder_label": vocoder_label,
        "vocoder_n": vocoder_n,
        "sample_rate": 16000,
        "file_name": file_name,
        "language_label": language,
        "spk_id":spk_id
    }

def getDataLoader(args,datapath,labelspath,batchsize,shuffle=True,num_workers=8,drop_last=True
                  ,dataset_collate_fn = None,data_labels = None,stage = "train",sampler=None):
    dataset = DetectDataset(args,
                            datapath=datapath,
                            labelspath=labelspath,
                            stage=stage,
                            collate_fn=dataset_collate_fn,
                            data_labels = data_labels)
    synthesizer = dataset.get_synthesiers()

    dataloader = DataLoader(dataset=dataset,batch_size=batchsize,
                            shuffle=shuffle,
                            num_workers=num_workers,
                            drop_last=drop_last,
                            collate_fn=simple_collate_fn,
                            sampler=sampler,
                            )
    return dataloader

import numpy as np
import scipy.signal as signal

def mask_7_8k(audio: np.ndarray, sample_rate: int, order: int = 201) -> np.ndarray:
    nyquist = 0.5 * sample_rate
    low = 7000.0 / nyquist
    high = 8000.0 / nyquist

    freq = [0.0, low, high, 1.0]
    gain = [1.0, 0.0, 0.0, 1.0]

    h = signal.firwin2(order, freq, gain, window='hamming')
    filtered = signal.filtfilt(h, 1.0, audio)
    # 强制转为 C 连续的正步幅数组
    return np.ascontiguousarray(filtered)

class DetectDataset(Dataset):
    def __init__(self, args , datapath,labelspath,stage="train",collate_fn = None,data_labels = None):
        # Generate synthetic data similar to your example
        self.datapath = datapath
        self.stage = stage
        self.args = args
        self.algo = args.algo 
        if data_labels == None:
            labels,synthesizer_num,synthesizers = getlabels(labelspath,collate_fn)
        else:
            labels,synthesizer_num,synthesizers = data_labels
        self.synthesizer_num = synthesizer_num
        self.filenames = labels['fns']
        self.synthesizerlabels = labels['synthesizers']
        self.synthesizers = synthesizers
        self.language = labels['language']
        self.spks = labels['spk_ids']
        print(synthesizers,set(self.language),len(self.filenames))
        print(args.exp_name)
        self.candidate_srs = [16000, 22050, 48000]
    def get_synthesiers(self):
        return self.synthesizer_num
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, index):
        fn = self.filenames[index]
        if '.' not in fn:
            fn = fn+'.wav'
        wav,sr = torchaudio.load(os.path.join(self.datapath,fn))
        
         # 处理标签
        label = 1 if self.synthesizers[self.synthesizerlabels[index]] == 0 else 0
        is_fake = (label == 0)   # 伪造样本
        
        # if self.stage == "train":
        #     wav = process_Rawboost_feature(wav.squeeze(0).numpy(),sr,self.args,self.algo)
        # # wav,sr,_ = audiospeed(wav,sr,self.min_speed,self.max_speed) # 数据增强
        #     wav = torch.from_numpy(wav).unsqueeze(0)
        wav = pad(wav).squeeze(0)
        sample = {
                "audio": wav,
                "label": label, # 合成器标签为0表示真实样本
                "vocoder_label": self.synthesizers[self.synthesizerlabels[index]],  # 合成器标签
                "vocoder_n":self.synthesizerlabels[index], # 合成器名称
                "sample_rate": sr, # sr =16000
                "file_name":fn,
                "language":0 if self.language[index] == 'en' else 1, # 英语标签为0
                "spk_id":self.spks[index]
            }
        return sample

import torch
import numpy as np
from tqdm import tqdm

def evaluate(model, dataloader, device):
    model.eval()
    bonadata = []
    spoofdata = []
    
    # --- 新增：用于计算准确率 ---
    all_preds_class = []
    all_labels_class = []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            audio = batch['audio'].to(device)
            # SSL特征提取
            # ssl_fea = sslModel(audio.squeeze(-1), mask=False, features_only=True)['layer_results']
            
            labels = batch['label'].to(device)
            # gat_t = aasist(audio)
            outputs,fea = model(audio) # 此时 outputs 是 LogSoftmax 结果
            
            # --- 修改点 1: 获取正类概率用于 EER ---
            # 假设类别1是目标类（根据你原来的代码 outputs[:, 1]）
            # 因为是 LogSoftmax，所以需要 exp() 转回概率值
            prob_class_1 = torch.exp(outputs[:, 1]).data.cpu().numpy().ravel()

            # --- 修改点 2: 获取预测类别用于准确率 ---
            # 直接取 log概率 最大的那个类别的索引 (0 或 1)
            _, predicted = torch.max(outputs.data, 1)
            
            for i in range(len(labels)):
                # 收集 EER 数据
                logit = prob_class_1[i]
                label_val = labels[i].item()
                
                if label_val == 1:
                    bonadata.append(logit)
                else:
                    spoofdata.append(logit)
                
                # 收集准确率数据
                all_preds_class.append(predicted[i].item())
                all_labels_class.append(label_val)

    # 1. 计算 EER
    bonadata = np.array(bonadata)
    spoofdata = np.array(spoofdata)
    eer, threshold = compute_eer(bonadata, spoofdata)

    # 2. 计算准确率
    all_preds_class = np.array(all_preds_class)
    all_labels_class = np.array(all_labels_class)
    
    # 通用准确率公式：(预测正确的数量) / (总数量)
    accuracy = np.sum(all_preds_class == all_labels_class) / len(all_labels_class)

    return eer, accuracy

def evaluate_synthesiser(dataloader,model,device,useeer = True):
    model.eval()
    bonadata = []
    algodict = {}
    preds = {}
    f = open("./scores/log.txt","a")
    fea_path = "/data/ys/res/SLS"
    os.makedirs(fea_path, exist_ok=True)
    num = 0
    vocoder_num = {}
    with torch.no_grad():
        for batch in tqdm(dataloader):
            num +=1
            if num > 100:
                break
            audio = batch['audio'].to(device)
            labels = batch['label'].to(device)
            algos = batch['vocoder_n']
            outputs,fea = model(audio)

            scores = (outputs[:, 1]).data.cpu().numpy().ravel()
            labels_np = labels.data.cpu().numpy()
            
            if useeer is False:
                pred = outputs[:,1].data.cpu() > threshold
                for i in range(len(labels)):
                    
                    algo = algos[i]
                    if labels[i] == 1:
                        algo = "bonafide"

                    f.write(f"{algo} : {outputs[i].cpu().numpy()} {pred[i]}\n")
                    if algo not in preds:
                        preds[algo] = {"correct":0,"total":0}
                    preds[algo]["correct"] += (pred[i] == labels[i]).item()
                    preds[algo]["total"] += 1
                continue
            outputs = (outputs[:, 1]).data.cpu().numpy().ravel()
            for i in range(len(labels)):
                logit = outputs[i]
                algo = algos[i]
                if labels[i] == 1:
                    bonadata.append(logit)
                    algo = "bonafide"
                if algo not in algodict:    
                    algodict[algo] = [] 
                algodict[algo].append(logit)
    f.close()        
    bonadata = np.array(bonadata)
    if useeer:
        eers = {}
        for algo in algodict.keys():
            if(algo == "bonafide"):
                continue
            algodict[algo] = np.array(algodict[algo])
            eer, threshold = compute_eer(bonadata, algodict[algo])
            eers[algo] = eer
        del algodict['bonafide']
        total_eer, threshold = compute_eer(bonadata, np.concatenate(list(algodict.values())))
        eers['total'] = total_eer
        eers["threshold_total"] = threshold
        return eers
    else:
        # 计算准确率
        accuracies = {}
        for algo, counts in preds.items():
            accuracy = counts["correct"] / counts["total"] if counts["total"] > 0 else 0
            accuracies[algo] = accuracy
            
        return accuracies    
    
# 按语言计算 EER
def evaluate_eer(dataloader,model,device,sslModel, aasist=None,useeer = True):
    
    model.eval()
    bonadata = []
    algodict = {}

    with torch.no_grad():
        for batch in tqdm(dataloader):
            
            audio = batch['audio'].to(device)
            labels = batch['label'].to(device)
            algos = batch['vocoder_n']

            # audio = sslModel(audio.squeeze(-1),mask=False, features_only=True)['layer_results']
            outputs,fea = model(audio)
            outputs = (outputs[:, 1]).data.cpu().numpy().ravel()
            for i in range(len(labels)):
                logit = outputs[i]
                algo = algos[i]
                if labels[i] == 1:
                    bonadata.append(logit)
                    algo = "bonafide"
                if algo not in algodict:    
                    algodict[algo] = [] 
                algodict[algo].append(logit)
    bonadata = np.array(bonadata)
    eers = {}
    for algo in algodict.keys():
        if(algo == "bonafide"):
            continue
        algodict[algo] = np.array(algodict[algo])
        eer, threshold = compute_eer(bonadata, algodict[algo])
        eers[algo] = eer
    del algodict['bonafide']
    total_eer, threshold = compute_eer(bonadata, np.concatenate(list(algodict.values())))
    eers['total'] = total_eer
    eers["threshold_total"] = threshold
    return eers
