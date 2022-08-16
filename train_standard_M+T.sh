#!/bin/bash
mkdir -p checkpoints
python -u train.py --name raft-big-M --stage movisubset --validation chairs --num_steps 100000 --batch_size 10 --lr 0.0004 --image_size 512 512 --wdecay 0.0001 --gpus 0
# python -u train.py --name raft-big-M+T --stage things --validation sintel --restore_ckpt checkpoints/raft-big-M.pth --num_steps 100000 --batch_size 6 --lr 0.000125 --image_size 400 720 --wdecay 0.0001 --gpus 0