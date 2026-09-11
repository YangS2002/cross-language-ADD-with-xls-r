#!/bin/sh
# CUDA_VISIBLE_DEVICES=0 python main.py \
#     --track=cl --lr=0.000001 --batch_size=6 --loss=WCE  --num_epochs=50 \
#     --protocols_path=/home/ys/SLS/database \
#     --database_path=/data/xls/Data_Source/ASVspoof2019/LA \

# CUDA_VISIBLE_DEVICES=1 python main.py \
#     --track=cl --lr=0.000001 --batch_size=3 --loss=WCE  --num_epochs=50 \
#     --protocols_path=/data/ys/petrichorwq-DECRO-dataset-6fc9884 \
#     --database_path=/data/ys/petrichorwq-DECRO-dataset-6fc9884 \


CUDA_VISIBLE_DEVICES=$1 python train.py --config ./config.yaml
# shutdown
    # --track=DECRO_en_train --lr=0.000001 --batch_size=8 --loss=WCE  --num_epochs=3 \
    # --protocols_path=/data/ys/petrichorwq-DECRO-dataset-6fc9884/en_train.txt \
    # --database_path=/data/ys/petrichorwq-DECRO-dataset-6fc9884/en_train \
# CUDA_VISIBLE_DEVICES=$1 python test.py --config /home/ys/SLS/config.yaml 