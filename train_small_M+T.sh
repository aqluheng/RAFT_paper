#!/bin/bash
WANDB_API_KEY=${APIKEY} WB_PROJECT=${PROJECT} WB_ID=0 python -u train.py --name raft-small-M --stage movisubset --validation chairs --num_steps 100000 --batch_size 10 --lr 0.0004 --image_size 512 512 --wdecay 0.0001 --gpus 0 --small
# python -u train.py --name raf-smallt-M+T --stage things --validation sintel --restore_ckpt checkpoints/raft-small-M.pth --num_steps 100000 --batch_size 6 --lr 0.000125 --image_size 400 720 --wdecay 0.0001 --gpus 0 --small