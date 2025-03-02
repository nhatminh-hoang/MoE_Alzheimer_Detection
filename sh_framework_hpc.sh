#!/bin/bash

#SBATCH --job-name=train_ADReSS2020
#SBATCH --nodes=1
#SBATCH --ntasks=1

#SBATCH --partition=dgx-small

#SBATCH --output=train_outs/small/out/%x.%j.out
#SBATCH --error=train_outs/small/errors/%x.%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=21013299@st.phenikaa-uni.edu.vn

conda init bash
source ~/.bashrc
conda activate minh_ml

python train.py --data_name ADReSS2020 --data_type text --text_type full --text_feature modernbert-base  --model Transformer --epochs 1000 --batch_size 256 --lr 1e-6 --hidden_size 16 --dropout 0.25 --early_stop no
python train.py --data_name ADReSS2020 --data_type text --text_type full --text_feature modernbert-base --model Transformer --epochs 1000 --batch_size 256 --lr 1e-6 --hidden_size 16 --dropout 0.5 --early_stop no
python train.py --data_name ADReSS2020 --data_type text --text_type full --text_feature modernbert-base --model Transformer --epochs 1000 --batch_size 256 --lr 1e-6 --hidden_size 16 --dropout 0.75 --early_stop no

python train.py --data_name ADReSS2020 --data_type text --text_type full --text_feature modernbert-base --model Transformer --epochs 1000 --batch_size 256 --lr 1e-6 --hidden_size 32 --dropout 0.25 --early_stop no
python train.py --data_name ADReSS2020 --data_type text --text_type full --text_feature modernbert-base --model Transformer --epochs 1000 --batch_size 256 --lr 1e-6 --hidden_size 32 --dropout 0.5 --early_stop no
python train.py --data_name ADReSS2020 --data_type text --text_type full --text_feature modernbert-base --model Transformer --epochs 1000 --batch_size 256 --lr 1e-6 --hidden_size 32 --dropout 0.75 --early_stop no

python train.py --data_name ADReSS2020 --data_type text --text_type full --text_feature modernbert-base --model Transformer --epochs 1000 --batch_size 256 --lr 1e-6 --hidden_size 64 --dropout 0.25 --early_stop no
python train.py --data_name ADReSS2020 --data_type text --text_type full --text_feature modernbert-base --model Transformer --epochs 1000 --batch_size 256 --lr 1e-6 --hidden_size 64 --dropout 0.5 --early_stop no
python train.py --data_name ADReSS2020 --data_type text --text_type full --text_feature modernbert-base --model Transformer --epochs 1000 --batch_size 256 --lr 1e-6 --hidden_size 64 --dropout 0.75 --early_stop no

python train.py --data_name ADReSS2020 --data_type text --text_type full --text_feature modernbert-base --model Transformer --epochs 1000 --batch_size 256 --lr 1e-6 --hidden_size 128 --dropout 0.25 --early_stop no
python train.py --data_name ADReSS2020 --data_type text --text_type full --text_feature modernbert-base --model Transformer --epochs 1000 --batch_size 256 --lr 1e-6 --hidden_size 128 --dropout 0.5 --early_stop no
python train.py --data_name ADReSS2020 --data_type text --text_type full --text_feature modernbert-base --model Transformer --epochs 1000 --batch_size 256 --lr 1e-6 --hidden_size 128 --dropout 0.75 --early_stop no

python train.py --data_name ADReSS2020 --data_type text --text_type full --text_feature modernbert-large  --model Transformer --epochs 1000 --batch_size 256 --lr 1e-6 --hidden_size 16 --dropout 0.25 --early_stop no
python train.py --data_name ADReSS2020 --data_type text --text_type full --text_feature modernbert-large --model Transformer --epochs 1000 --batch_size 256 --lr 1e-6 --hidden_size 16 --dropout 0.5 --early_stop no
python train.py --data_name ADReSS2020 --data_type text --text_type full --text_feature modernbert-large --model Transformer --epochs 1000 --batch_size 256 --lr 1e-6 --hidden_size 16 --dropout 0.75 --early_stop no

python train.py --data_name ADReSS2020 --data_type text --text_type full --text_feature modernbert-large --model Transformer --epochs 1000 --batch_size 256 --lr 1e-6 --hidden_size 32 --dropout 0.25 --early_stop no
python train.py --data_name ADReSS2020 --data_type text --text_type full --text_feature modernbert-large --model Transformer --epochs 1000 --batch_size 256 --lr 1e-6 --hidden_size 32 --dropout 0.5 --early_stop no
python train.py --data_name ADReSS2020 --data_type text --text_type full --text_feature modernbert-large --model Transformer --epochs 1000 --batch_size 256 --lr 1e-6 --hidden_size 32 --dropout 0.75 --early_stop no

python train.py --data_name ADReSS2020 --data_type text --text_type full --text_feature modernbert-large --model Transformer --epochs 1000 --batch_size 256 --lr 1e-6 --hidden_size 64 --dropout 0.25 --early_stop no
python train.py --data_name ADReSS2020 --data_type text --text_type full --text_feature modernbert-large --model Transformer --epochs 1000 --batch_size 256 --lr 1e-6 --hidden_size 64 --dropout 0.5 --early_stop no
python train.py --data_name ADReSS2020 --data_type text --text_type full --text_feature modernbert-large --model Transformer --epochs 1000 --batch_size 256 --lr 1e-6 --hidden_size 64 --dropout 0.75 --early_stop no

python train.py --data_name ADReSS2020 --data_type text --text_type full --text_feature modernbert-large --model Transformer --epochs 1000 --batch_size 256 --lr 1e-6 --hidden_size 128 --dropout 0.25 --early_stop no
python train.py --data_name ADReSS2020 --data_type text --text_type full --text_feature modernbert-large --model Transformer --epochs 1000 --batch_size 256 --lr 1e-6 --hidden_size 128 --dropout 0.5 --early_stop no
python train.py --data_name ADReSS2020 --data_type text --text_type full --text_feature modernbert-large --model Transformer --epochs 1000 --batch_size 256 --lr 1e-6 --hidden_size 128 --dropout 0.75 --early_stop no
