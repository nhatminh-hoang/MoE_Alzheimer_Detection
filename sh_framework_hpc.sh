#!/bin/bash

#SBATCH --job-name=train_ADReSS2020
#SBATCH --nodes=1
#SBATCH --ntasks=1

#SBATCH --partition=dgx-small
#SBATCH --gres=gpu:1
#SBATCH --output=train_outs/small/out/%x.%j.out
#SBATCH --error=train_outs/small/errors/%x.%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=21013299@st.phenikaa-uni.edu.vn

pip install -r requirements.txt

python train.py --data_name ADReSS2020 --wave_type full --feature_type MFCC --model Transformer --epochs 1000 --batch_size 256 --lr 1e-3 --hidden_size 128 --dropout 0.25 --early_stop no
python train.py --data_name ADReSS2020 --wave_type chunk --feature_type MFCC --model Transformer --epochs 1000 --batch_size 256 --lr 1e-3 --hidden_size 32 --dropout 0.25 --early_stop no
python train.py --data_name ADReSS2020 --wave_type chunk --feature_type MFCC --model Transformer --epochs 1000 --batch_size 256 --lr 1e-3 --hidden_size 64 --dropout 0.25 --early_stop no
python train.py --data_name ADReSS2020 --wave_type chunk --feature_type MFCC --model Transformer --epochs 1000 --batch_size 256 --lr 1e-3 --hidden_size 128 --dropout 0.25 --early_stop no
python train.py --data_name ADReSS2020 --wave_type chunk --feature_type MFCC --model Transformer --epochs 1000 --batch_size 256 --lr 1e-3 --hidden_size 256 --dropout 0.25 --early_stop no
python train.py --data_name ADReSS2020 --wave_type chunk --feature_type MFCC --model Transformer --epochs 1000 --batch_size 256 --lr 1e-3 --hidden_size 512 --dropout 0.25 --early_stop no