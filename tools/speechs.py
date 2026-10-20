import librosa
import os
from tqdm import tqdm
# 统计某个文件夹下面所有的wav文件的总时长和文件数

wavpath = '/data/xls/Data_Source/ASVspoof2021/ASVspoof2021_LA_eval/wav'
totaltime =0
totalwav = 0
for speaker in tqdm(os.listdir(wavpath)):
    if os.path.isdir(os.path.join(wavpath,speaker)):
        for fn in os.listdir(os.path.join(wavpath,speaker)):
            if fn.split('.')[-1] == 'lab':
                continue
            wav,sr = librosa.load(os.path.join(wavpath,speaker,fn))
            totaltime += wav.shape[0]/sr
            totalwav += 1
    else:
        if speaker.split('.')[-1] == 'wav':
            wav,sr = librosa.load(os.path.join(wavpath,speaker))
            totaltime += wav.shape[0]/sr
            totalwav += 1
    
print(f"总时长：{totaltime}s（{totaltime/60} min），总文件数：{totalwav}，文件路径：{wavpath}")