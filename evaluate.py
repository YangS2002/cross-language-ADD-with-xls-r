# tortoise EER评估
# 真实语音来自LJSpeech，伪造语音来自tortoise-standared
import eval_metrics_DF as em
import pandas
import numpy as np
def readscore(filepath):
    data = open(filepath,'r').read().split('\n')
    data = data[:-2]
    data = [float(i) for i in data]
    return pandas.array(data)

def readfile(path,separator=' '):
    # 读取文件，并按‘\n’分行，然后每行按分隔符切片，默认空格做分隔符
    ret = []
    with open(path,'r') as f:
        labels = f.read()
        labels = labels.split('\n')
        for line in labels:
            line = line.split(separator)
            if len(line)<2: # 去除空行
                continue
            ret.append(line)
        f.close()
    return ret

def eval_tortoise():
    # scorespath = '/home/ys/SLS/scores/test/tortoise.txt'
    scorespath = '/home/ys/SLS/scores/test/crosslingual_chattts.txt'
    with open(scorespath,'r') as f:
        scores = f.read()
        scores = scores.split('\n') # 分行
        bona_data = []
        spoof_data = []
        # print(scores)
        for score in scores:
            score = score.split(' ')
            if len(score)!=2:
                break
            name = score[0]
            score = float(score[1])
            if 'spoof' in name:
                spoof_data.append(score)
            elif 'bona' in name:
                bona_data.append(score)
        bona_data = pandas.array(bona_data)
        spoof_data = pandas.array(spoof_data)
        eer = em.compute_eer(bona_data,spoof_data)[0]
        eer = eer*100
        print('eer: %.2f'%(eer))

def eval_CrossLingual():
    bona_filepath = './scores/test/CrossLingual_bona_socres.txt'
    spoof_filepath = './scores/test/CrossLingual_spoof_scores.txt'
    bona_data = readscore(bona_filepath)
    spoof_data = readscore(spoof_filepath)
    eer = em.compute_eer(bona_data,spoof_data)[0]
    eer = eer*100
    print(eer)

def cross_lingual_Score_DECRO():
    labels_path = "/data/ys/petrichorwq-DECRO-dataset-6fc9884/ch_eval.txt"
    scores_path = "/home/ys/SLS/scores/test/crosslingual_DECRO_ch.txt"
    labels = readfile(labels_path) 
    name_label = {labels[i][1]:labels[i][-1] for i in range(len(labels))}
    name_score = readfile(scores_path)
    bona_data = []
    spoof_data = []
    for ns in name_score:
        scores = np.exp(float(ns[1]))
        wavname = ns[0].split('.')[0]
        if 'bona' in name_label[wavname]:
            bona_data.append(scores)
        if 'spoof' in name_label[wavname]:
            spoof_data.append(scores)
    bona_data = pandas.array(bona_data)
    spoof_data = pandas.array(spoof_data)
    eer = em.compute_eer(bona_data,spoof_data)[0]
    eer = eer*100
    print(eer)
              
def eval_inthewild():
    scroes_path = "./scores/DF_retrain/in-the-wild.txt"
    label_path = "/data/ys/release_in_the_wild/meta.csv"
    labels = readfile(label_path,',')[1:] # 文件名，说话人，label
    filename_labels = {}
    for line in labels:
        filename = line[0]
        label = line[2]
        filename_labels[filename] = label
    scores = readfile(scroes_path,' ')
    bona_data = []
    spoof_data = []
    for line in scores:
        if len(line) <2:
            continue
        filename = line[0]
        score = line[1]
        if 'bona' in filename_labels[filename]:
            bona_data.append(float(score))
        elif 'spoof' in filename_labels[filename]:
            spoof_data.append(float(score))
    bona_data = pandas.array(bona_data)
    spoof_data = pandas.array(spoof_data)
    eer = em.compute_eer(bona_data,spoof_data)[0]
    eer = eer*100
    print(eer)

def eval_ASVSpoof2021_DF():
    scores_path = "/home/ys/SLS/scores/DF_retrain/ASVSpoof2021_DF_best_model.txt"
    labels_path = "/data/xls/Data_Source/ASVspoof2021/keys/DF/CM/ASVspoof2021.DF.cm.eval.trl.txt"
    # LA_0023 DF_E_2000011 - A14 spoof 空格切分，第二个是文件名，倒数第一个是label
    labels = readfile(labels_path,' ')
    name_labels = {}
    for line in labels:
        name_labels[line[1]] = line[-1]
    name_scores = readfile(scores_path,' ')
    bona_data = []
    spoof_data = []
    for score in name_scores:
        label = name_labels[score[0]]
        if "bona" in label:
            bona_data.append(float(score[1]))
        elif "spoof" in label:
            spoof_data.append(float(score[1]))
    bona_data = pandas.array(bona_data)
    spoof_data = pandas.array(spoof_data)
    eer = em.compute_eer(bona_data,spoof_data)[0]
    eer = eer*100
    print(eer)


if __name__ == '__main__':
    # 写入的得分数据都是两个元素，第一个是文件名，第二个是得分
    # eval_CrossLingual()
    cross_lingual_Score_DECRO()
    # eval_tortoise()
    # eval_inthewild()
    # eval_ASVSpoof2021_DF()
