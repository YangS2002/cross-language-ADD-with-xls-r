import os
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
import librosa
from torch.utils.data import Dataset
from RawBoost import ISD_additive_noise,LnL_convolutive_noise,SSI_additive_noise,normWav
from random import randrange
import random

from tools.utils import getTrainData,getCLTrainData
from torch.utils.data import DataLoader

def genSpoof_list2019(dir_meta, is_train=False, is_eval=False):
    d_meta = {}
    file_list = []
    with open(dir_meta, 'r') as f:
        l_meta = f.readlines()

    if (is_eval):
        for line in l_meta:
            key = line.strip()
            # _, key, _, _, _ = line.strip().split()
            file_list.append(key)
        return file_list

def genSpoof_list(dir_meta, is_train=False, is_eval=False):
    
    d_meta = {}
    file_list=[]
    with open(dir_meta, 'r') as f:
         l_meta = f.readlines()

    if (is_train):
        for line in l_meta:
             _,key,_,_,label = line.strip().split()
             
             file_list.append(key)
             d_meta[key] = 1 if label == 'bonafide' else 0
        return d_meta,file_list
    
    elif(is_eval):
        for line in l_meta:
            key= line.strip()
            #_, key, _, _, _ = line.strip().split()
            file_list.append(key)
        return file_list
    else:
        for line in l_meta:
             _,key,_,_,label = line.strip().split()
             
             file_list.append(key)
             d_meta[key] = 1 if label == 'bonafide' else 0
        return d_meta,file_list

def genDataloader(config,args,datas_path,labels_path,is_train=False):
    if is_train == True:
        shuffle = True
        drop_last = True
    else:
        shuffle = False
        drop_last = False
    
    if config['train_tag'] == "normal":
        filenames_labels,filenames = getTrainData(labels_path)
        train_set = DatasetTrain(wavsDir=datas_path,
                                wavnames=filenames,
                                labels=filenames_labels,
                                args = args,
                                )
        train_loader = DataLoader(train_set, batch_size=config['batch_size'],
                                num_workers=config['num_workers'], 
                                shuffle=shuffle,drop_last = drop_last)
        del filenames,filenames_labels
        return train_loader
    elif config['train_tag'] == "cl":
        # multi-language
        ch_filenames_labels,ch_filenames,en_filenames_labels,en_filenames = getCLTrainData(
            config['train_ch_label_path'],
            config['train_label_path']
        )
        ch_train_set = DatasetTrain(wavsDir=config['train_ch_data_path'],
                                wavnames=ch_filenames,
                                labels=ch_filenames_labels,
                                args = args,
                                )
        ch_train_loader = DataLoader(ch_train_set, batch_size=config['batch_size'],
                                num_workers=config['num_workers'], 
                                shuffle=True,drop_last = True)
        en_train_set = DatasetTrain(wavsDir=config['train_data_path'],
                                wavnames=en_filenames,
                                labels=en_filenames_labels,
                                args = args,
                                )
        en_train_loader = DataLoader(en_train_set, batch_size=config['batch_size'],
                                num_workers=config['num_workers'], 
                                shuffle=True,drop_last = True)
        del ch_filenames_labels,ch_filenames,en_filenames_labels,en_filenames
        return en_train_loader,ch_train_loader

def pad(x, max_len=64600):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    # need to pad
    num_repeats = int(max_len / x_len)+1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    return padded_x	
			

class Dataset_crossluanguage_train(Dataset):
    def __init__(self,args,list_IDs, labels, base_dir,other_language_base_dir,algo):
        '''self.list_IDs	: list of strings (each string: utt key),
            self.labels      : dictionary (key: utt key, value: label integer)'''
            
        self.list_IDs = list_IDs
        self.labels = labels
        self.base_dir = base_dir
        self.other_language_base_dir = other_language_base_dir
        self.algo=algo
        self.args=args
        self.cut=64600 # take ~4 sec audio (64600 samples)

    def __len__(self):
        return len(self.list_IDs)


    def __getitem__(self, index):
        utt_id = self.list_IDs[index]
        if 'LA' in utt_id: 
            X,fs = librosa.load(os.path.join(self.base_dir,'flac',utt_id+'.flac'), sr=16000) 
        else:
             X,fs = librosa.load(os.path.join(self.other_language_base_dir,'ch_train',utt_id+'.wav'), sr=16000) 
        Y=process_Rawboost_feature(X,fs,self.args,self.algo)
        X_pad= pad(Y,self.cut)
        x_inp= Tensor(X_pad)
        target = self.labels[utt_id]
        
        return x_inp, target

class Dataset_ASVspoof2019_train(Dataset):
    def __init__(self,args,list_IDs, labels, base_dir,algo):
        '''self.list_IDs	: list of strings (each string: utt key),
            self.labels      : dictionary (key: utt key, value: label integer)'''
            
        self.list_IDs = list_IDs
        self.labels = labels
        self.base_dir = base_dir
        self.algo=algo
        self.args=args
        self.cut=64600 # take ~4 sec audio (64600 samples)

    def __len__(self):
        return len(self.list_IDs)


    def __getitem__(self, index):
            
        utt_id = self.list_IDs[index] # 文件名
        X,fs = librosa.load(self.base_dir+'flac/'+utt_id+'.flac', sr=16000) 
        
        Y=process_Rawboost_feature(X,fs,self.args,self.algo)
        X_pad= pad(Y,self.cut)
        x_inp= Tensor(X_pad)
        target = self.labels[utt_id]
        
        return x_inp, target
            
class DatasetTrain(Dataset):
    def __init__(self,wavnames, labels, wavsDir,args):
        '''self.list_IDs	: list of strings (each string: utt key),
            self.labels      : dictionary (key: utt key, value: label integer)'''
            
        self.wavnames = wavnames # 文件名数组
        self.labels = labels # 文件名——标签
        self.wavsDir = wavsDir # 语音的文件夹
        self.cut=64600 # take ~4 sec audio (64600 samples)
        self.algo = args.algo
        self.args = args

    def __len__(self):
        return len(self.wavnames)


    def __getitem__(self, index):
            
        wavname = self.wavnames[index] # 文件名
        X,fs = librosa.load(os.path.join(self.wavsDir,wavname), sr=16000) 
        
        Y=process_Rawboost_feature(X,fs,self.args,self.algo)
        X_pad= pad(X,self.cut)
        x_inp= Tensor(X_pad)
        target = self.labels[wavname]
        
        return x_inp, target        

class Dataset_ASVspoof2021_eval(Dataset):
    def __init__(self, list_IDs, base_dir):
        '''self.list_IDs	: list of strings (each string: utt key),
            '''
            
        self.list_IDs = list_IDs
        self.base_dir = base_dir
        self.cut=64600 # take ~4 sec audio (64600 samples)

    def __len__(self):
        return len(self.list_IDs)


    def __getitem__(self, index):
            
        utt_id = self.list_IDs[index]
        X, fs = librosa.load(self.base_dir+'/flac/'+utt_id+'.flac', sr=16000)
        X_pad = pad(X,self.cut)
        x_inp = Tensor(X_pad)
        return x_inp,utt_id

class DatasetEval(Dataset):
    def __init__(self,fileNames,base_dir) :
        self.fileNames = fileNames
        self.base_dir = base_dir
        self.cut = 64600 # ?? 为什么要填充到64600
    def __len__(self):
        return len(self.fileNames)
    
    def __getitem__(self,index):
        filename = self.fileNames[index]
        X,fs = librosa.load(os.path.join(self.base_dir,filename),sr = 16000)
        X_pad = pad(X,self.cut)
        X_inp = Tensor(X_pad)
        return X_inp,filename  

def process_Rawboost_feature(feature, sr,args,algo):
    
    # Data process by Convolutive noise (1st algo)
    if algo==1:

        feature =LnL_convolutive_noise(feature,args.N_f,args.nBands,args.minF,args.maxF,args.minBW,args.maxBW,args.minCoeff,args.maxCoeff,args.minG,args.maxG,args.minBiasLinNonLin,args.maxBiasLinNonLin,sr)
                            
    # Data process by Impulsive noise (2nd algo)
    elif algo==2:
        
        feature=ISD_additive_noise(feature, args.P, args.g_sd)
                            
    # Data process by coloured additive noise (3rd algo)
    elif algo==3:
        
        feature=SSI_additive_noise(feature,args.SNRmin,args.SNRmax,args.nBands,args.minF,args.maxF,args.minBW,args.maxBW,args.minCoeff,args.maxCoeff,args.minG,args.maxG,sr)
    
    # Data process by all 3 algo. together in series (1+2+3)
    elif algo==4:
        
        feature =LnL_convolutive_noise(feature,args.N_f,args.nBands,args.minF,args.maxF,args.minBW,args.maxBW,
                 args.minCoeff,args.maxCoeff,args.minG,args.maxG,args.minBiasLinNonLin,args.maxBiasLinNonLin,sr)                         
        feature=ISD_additive_noise(feature, args.P, args.g_sd)  
        feature=SSI_additive_noise(feature,args.SNRmin,args.SNRmax,args.nBands,args.minF,
                args.maxF,args.minBW,args.maxBW,args.minCoeff,args.maxCoeff,args.minG,args.maxG,sr)                 

    # Data process by 1st two algo. together in series (1+2)
    elif algo==5:
        
        feature =LnL_convolutive_noise(feature,args.N_f,args.nBands,args.minF,args.maxF,args.minBW,args.maxBW,
                 args.minCoeff,args.maxCoeff,args.minG,args.maxG,args.minBiasLinNonLin,args.maxBiasLinNonLin,sr)                         
        feature=ISD_additive_noise(feature, args.P, args.g_sd)                
                            

    # Data process by 1st and 3rd algo. together in series (1+3)
    elif algo==6:  
        
        feature =LnL_convolutive_noise(feature,args.N_f,args.nBands,args.minF,args.maxF,args.minBW,args.maxBW,
                 args.minCoeff,args.maxCoeff,args.minG,args.maxG,args.minBiasLinNonLin,args.maxBiasLinNonLin,sr)                         
        feature=SSI_additive_noise(feature,args.SNRmin,args.SNRmax,args.nBands,args.minF,args.maxF,args.minBW,args.maxBW,args.minCoeff,args.maxCoeff,args.minG,args.maxG,sr) 

    # Data process by 2nd and 3rd algo. together in series (2+3)
    elif algo==7: 
        
        feature=ISD_additive_noise(feature, args.P, args.g_sd)
        feature=SSI_additive_noise(feature,args.SNRmin,args.SNRmax,args.nBands,args.minF,args.maxF,args.minBW,args.maxBW,args.minCoeff,args.maxCoeff,args.minG,args.maxG,sr) 
   
    # Data process by 1st two algo. together in Parallel (1||2)
    elif algo==8:
        
        feature1 =LnL_convolutive_noise(feature,args.N_f,args.nBands,args.minF,args.maxF,args.minBW,args.maxBW,
                 args.minCoeff,args.maxCoeff,args.minG,args.maxG,args.minBiasLinNonLin,args.maxBiasLinNonLin,sr)                         
        feature2=ISD_additive_noise(feature, args.P, args.g_sd)

        feature_para=feature1+feature2
        feature=normWav(feature_para,0)  #normalized resultant waveform
 
    # original data without Rawboost processing           
    else:
        
        feature=feature
    
    return feature
