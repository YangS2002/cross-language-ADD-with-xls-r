#!/bin/sh
# 评估 in the wild 数据集
# CUDA_VISIBLE_DEVICES=1 python main.py   --track=In-the-Wild --is_eval --eval \
#                                         --model_path=/home/ys/SLS/models/model_DF_WCE_50_6_1e-06/epoch_4.pth \
#                                         --protocols_path=./database/ASVspoof_DF_cm_protocols/in_the_wild.eval.txt \
#                                         --database_path=/data/ys/release_in_the_wild/ \
#                                         --eval_output=/home/ys/SLS/scores/DF_retrain/in-the-wild.txt

# 评估 LJSpeech-Tortoise 语音合成 数据集
# 生成待评估数据
# echo "gen evaluation data"
# python test.py tortoise
# CUDA_VISIBLE_DEVICES=1 python main.py   --track=Tortoise --is_eval --eval \
#                                         --model_path=/home/ys/SLS/models/MMpaper_model.pth \
#                                         --protocols_path=./database/tortoise/test.eval.txt \
#                                         --database_path=/home/ys/SLS/tortoise \
#                                         --eval_output=/home/ys/SLS/scores/test/tortoise.txt

# echo "start sparate bona and spoof data"
# cd /home/ys/SLS/scores/test/
# ./run.sh
# echo "start evaluate!"
# cd /home/ys/SLS/
# python evaluate_tortoise.py
# rm /home/ys/SLS/scores/test/tortoise.txt

# 评估指标计算
# python evaluate_in_the_wild.py /home/ys/SLS/scores/test/in-the-wild.txt \
#                                 /home/ys/SLS/keys/ \
#                                 eval

# # 评估跨语言检测能力
# echo "evaluate cross-language deepfake detect"
# python test.py crosslingual
# CUDA_VISIBLE_DEVICES=1 python main.py   --track=Tortoise --is_eval --eval \
#                                         --model_path=/home/ys/SLS/models/MMpaper_model.pth \
#                                         --protocols_path=./database/crosslingual/crossLingual_test.eval.txt \
#                                         --database_path=/home/ys/datasets/CVoiceFake_small/zh-CN \
#                                         --eval_output=/home/ys/SLS/scores/test/crosslingual.txt

#评估跨语言检测能力-DECRO-中文子数据集
# echo "evaluate cross-language deepfake detection on DECRO"
# # python ./tools/test.py crosslingual_DECRO
# CUDA_VISIBLE_DEVICES=2 python main.py   --track=eval --is_eval --eval \
#                                         --model_path=/home/ys/SLS/models/MMpaper_model.pth\
#                                         --protocols_path=./database/crosslingual_DECRO/crossLingual_test.eval.txt \
#                                         --database_path=/data/ys/petrichorwq-DECRO-dataset-6fc9884/ch_eval \
#                                         --eval_output=/home/ys/SLS/scores/test/crosslingual_DECRO_ch.txt

# 评估跨语言检测能力-DECRO-英语子数据集
# echo "evaluate cross-language deepfake detection on DECRO English"
# # python ./tools/test.py crosslingual_DECRO 
# CUDA_VISIBLE_DEVICES=2 python main.py   --track=eval --is_eval --eval \
#                                         --model_path=/home/ys/SLS/models/MMpaper_model.pth \
#                                         --protocols_path=./database/crosslingual_DECRO/crossLingual_test_en.eval.txt \
#                                         --database_path=/data/ys/petrichorwq-DECRO-dataset-6fc9884/en_eval \
#                                         --eval_output=/home/ys/SLS/scores/DECRO/DECRO_en_test.txt

# 评估跨语言检测能力-chattts-中文
# echo "evaluate cross-language deepfake detection,wavs synthesised with chattts chaniese"
# python ./tools/test.py crosslingual_chattts
# CUDA_VISIBLE_DEVICES=0 python main.py   --track=Tortoise --is_eval --eval \
#                                         --model_path=/home/ys/SLS/models/MMpaper_model.pth \
#                                         --protocols_path=/home/ys/SLS/database/crosslingual_chattts/crossLingual_chattts_test.eval.txt \
#                                         --database_path=/home/ys/datasets/chattts \
#                                         --eval_output=/home/ys/SLS/scores/test/crosslingual_chattts.txt

# 评估ASVSpoof2021_DF数据集
# echo "evaluate deepfake detection,wavs from ASVSpoof2021 DF"
# CUDA_VISIBLE_DEVICES=2 python main.py   --track=DF --is_eval --eval \
#                                         --model_path=/home/ys/SLS/models/MMpaper_model.pth \
#                                         --protocols_path=database/ASVspoof_DF_cm_protocols/ASVspoof2021.DF.cm.eval.trl.txt \
#                                         --database_path=/data/xls/Data_Source/ASVspoof2021/ASVspoof2021_DF_eval \
#                                         --eval_output=scores/test/ASVSpoof2021_DF.txt
# echo "evaluate cross-language deepfake detection on DECRO"
# # python ./tools/test.py crosslingual_DECRO
# CUDA_VISIBLE_DEVICES=1 python main.py   --track=eval --is_eval --eval \
#                                         --model_path=/home/ys/SLS/models/DECRO_en/epoch3_model_0.0004614895177380745.pth\
#                                         --protocols_path=./database/crosslingual_DECRO/crossLingual_test_ch.eval.txt \
#                                         --database_path=/data/ys/petrichorwq-DECRO-dataset-6fc9884/ch_eval \
#                                         --eval_output=/home/ys/SLS/scores/test/crosslingual_DECRO_ch.txt

# CUDA_VISIBLE_DEVICES=$1 python train.py \
#     --config /home/ys/SLS/config.yaml
    
CUDA_VISIBLE_DEVICES=$1 python test.py --config ./config.yaml