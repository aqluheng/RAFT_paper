from __future__ import print_function, division
import sys
sys.path.append('core')

import argparse
import os
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader
from raft import RAFT
import evaluate
import datasets

try:
    import wandb
except:
    pass


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class Logger:
    def __init__(self):
        pass
        
    def write_dict(self, results, step):
        wandb.log(results, step=step)

    def close(self):
        pass


def evaluateAllStep(args):

    model = nn.DataParallel(RAFT(args), device_ids=args.gpus)
    print("Parameter Count: %d" % count_parameters(model))

    model.cuda()
    model.eval()

    logger = Logger()

    if args.enable_wandb:
        wandb_id = os.environ.get("NGC_JOB_ID", None)
#         wandb_id = "3268105"
        env_id = os.environ.get("WB_ID", None)
        if env_id is not None:
            if wandb_id is not None:
                wandb_id += f'_{env_id}'
            else:
                wandb_id = env_id
        wandb.init(project=os.environ.get('WB_PROJECT', None), id=wandb_id+"_"+args.name+"_"+str(args.iters), name=args.name+str(args.iters), config=vars(args))
        wandb.define_metric("final", summary="min")
        wandb.define_metric("final-epe", summary="min")

    for total_steps in range(4999, 99999+1, 5000):
        PATH = 'checkpoints/%d_%s.pth' % (total_steps+1, args.name)
        model.load_state_dict(torch.load(PATH))
        results = {}
        for val_dataset in args.validation:
            if val_dataset == 'chairs':
                results.update(evaluate.validate_chairs(model.module, iters=args.iters))
            elif val_dataset == 'sintel':
                results.update(evaluate.validate_sintel(model.module, iters=args.iters))
            elif val_dataset == 'kitti':
                results.update(evaluate.validate_kitti(model.module, iters=args.iters))
        logger.write_dict(results, total_steps)

    logger.close()
    return PATH


# +
# args = argparse.Namespace()
# args.name = "raft-big-movi-autoflowAug"
# args.iters = 32
# args.gpus = [0]
# args.small = False
# args.enable_wandb = False
# args.mixed_precision = False
# args.validation = ["sintel"]

# evaluateAllStep(args)
# -

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='raft', help="name your experiment")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--validation', type=str, nargs='+')

    parser.add_argument('--mixed_precision', action='store_true', help="use mixed precision")
    parser.add_argument('--iters', type=int, required=True)
    parser.add_argument('--enable_wandb', action='store_true', help='enable wandb to trace experiments')
    parser.add_argument('--gpus', type=int, nargs='+', default=[0])
    args = parser.parse_args()

    torch.manual_seed(1234)
    np.random.seed(1234)

    evaluateAllStep(args)
    if args.enable_wandb:
        wandb.finish()


