#!/bin/bash
mkdir -p checkpoints
# python -u train.py --name raft-small-M --stage movisubset --validation chairs --num_steps 100000 --batch_size 10 --lr 0.0004 --image_size 512 512 --wdecay 0.0001 --gpus 0 --small
python -u train.py --name raf-smallt-M+T --stage things --validation sintel --restore_ckpt checkpoints/raft-small-M.pth --num_steps 100000 --batch_size 6 --lr 0.000125 --image_size 400 720 --wdecay 0.0001 --gpus 0 --small
python train.py --name raft-tradeoff-SD+V --iters 12 --stage virtualkitti2 --validation kitti --num_steps 40000 --batch_size 12 --lr 0.0004 --image_size 288 960 --wdecay 0.00001 --small --restore_ckpt checkpoints/raft-tradeoff-SD.pth
python train.py --name raft-tradeoff-SD+T+V --iters 12 --stage virtualkitti2 --validation kitti --num_steps 40000 --batch_size 12 --lr 0.0004 --image_size 288 960 --wdecay 0.00001 --small --restore_ckpt checkpoints/raft-tradeoff-SD+T.pth

