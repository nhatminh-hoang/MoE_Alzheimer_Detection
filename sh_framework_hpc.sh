#!/bin/bash

#SBATCH --job-name=train_ADReSS2020
#SBATCH --nodes=1
#SBATCH --ntasks=1

#SBATCH --partition=dgx-small

#SBATCH --output=train_outs/small/out/%x.%j.out
#SBATCH --error=train_outs/small/errors/%x.%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=21013299@st.phenikaa-uni.edu.vn

python train.py --data_name ADReSS2020 --wave_type full --feature_type MFCC --model CNN --epochs 1000 --batch_size 256 --lr 1e-3 --hidden_size 128 --dropout 0.5 --early_stop no
python train.py --data_name ADReSS2020 --wave_type full --feature_type MFCC --model LSTM --epochs 1000 --batch_size 256 --lr 1e-3 --hidden_size 128 --dropout 0.5 --early_stop no
python train.py --data_name ADReSS2020 --wave_type full --feature_type MFCC --model BiLSTM --epochs 1000 --batch_size 256 --lr 1e-3 --hidden_size 128 --dropout 0.5 --early_stop no
python train.py --data_name ADReSS2020 --wave_type full --feature_type MFCC --model Transformer --epochs 1000 --batch_size 256 --lr 1e-3 --hidden_size 128 --dropout 0.5 --early_stop no
python train.py --data_name ADReSS2020 --data_type text --model Transformer --epochs 1000 --batch_size 256 --lr 1e-3 --hidden_size 128 --dropout 0.5 --early_stop no
