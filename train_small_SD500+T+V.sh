#!/bin/bash
pip install wandb scikit-image
APIKEY="09c6e5e2dbd5e78ff164ef5477b02778d42ce95f"
PROJECT="RAFT"

WANDB_API_KEY=${APIKEY} WB_PROJECT=${PROJECT} python -u train.py --name raft-small-SD --stage smalldense500 --validation sintel --num_steps 100000 --batch_size 10 --lr 0.0004 --image_size 512 512 --wdecay 0.0001 --gpus 0 --small --enable_wandb 
WANDB_API_KEY=${APIKEY} WB_PROJECT=${PROJECT} python -u train.py --name raft-small-SD+T --stage things --validation sintel --restore_ckpt checkpoints/raft-small-SD.pth --num_steps 100000 --batch_size 6 --lr 0.000125 --image_size 400 720 --wdecay 0.0001 --gpus 0 --small --enable_wandb 
WANDB_API_KEY=${APIKEY} WB_PROJECT=${PROJECT} python -u train.py --name raft-small-SD+V --iters 12 --stage virtualkitti2 --validation kitti --num_steps 40000 --batch_size 12 --lr 0.0004 --image_size 288 960 --wdecay 0.00001 --small --enable_wandb  --restore_ckpt checkpoints/raft-small-SD.pth
WANDB_API_KEY=${APIKEY} WB_PROJECT=${PROJECT} python -u train.py --name raft-small-SD+T+V --iters 12 --stage virtualkitti2 --validation kitti --num_steps 40000 --batch_size 12 --lr 0.0004 --image_size 288 960 --wdecay 0.00001 --small --enable_wandb  --restore_ckpt checkpoints/raft-small-SD+T.pth

