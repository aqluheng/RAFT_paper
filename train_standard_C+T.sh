#!/bin/bash
mkdir -p checkpoints
python -u train.py --name raft-big-C --stage chairs --validation chairs --num_steps 100000 --batch_size 10 --lr 0.0004 --image_size 368 496 --wdecay 0.0001 --gpus 0
python -u train.py --name raft-big-C+T --stage things --validation sintel --restore_ckpt checkpoints/raft-big-C.pth --num_steps 100000 --batch_size 6 --lr 0.000125 --image_size 400 720 --wdecay 0.0001 --gpus 0

