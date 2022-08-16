import sys
sys.path.append('core')

from PIL import Image
import argparse
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import cv2

import datasets
from utils import flow_viz
from utils import frame_utils
from utils import sintel_io
from utils import ofutil

from raft import RAFT
from utils.utils import InputPadder, forward_interpolate


@torch.no_grad()
def create_sintel_submission(model, iters=32, warm_start=False, output_path='sintel_submission'):
    """ Create submission for the Sintel leaderboard """
    model.eval()
    for dstype in ['clean', 'final']:
        test_dataset = datasets.MpiSintel(split='test', aug_params=None, dstype=dstype)
        
        flow_prev, sequence_prev = None, None
        for test_id in range(len(test_dataset)):
            image1, image2, (sequence, frame) = test_dataset[test_id]
            if sequence != sequence_prev:
                flow_prev = None
            
            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1[None].cuda(), image2[None].cuda())

            flow_low, flow_pr = model(image1, image2, iters=iters, flow_init=flow_prev, test_mode=True)
            flow = padder.unpad(flow_pr[0]).permute(1, 2, 0).cpu().numpy()

            if warm_start:
                flow_prev = forward_interpolate(flow_low[0])[None].cuda()
            
            output_dir = os.path.join(output_path, dstype, sequence)
            output_file = os.path.join(output_dir, 'frame%04d.flo' % (frame+1))

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            frame_utils.writeFlow(output_file, flow)
            sequence_prev = sequence


@torch.no_grad()
def create_kitti_submission(model, iters=24, output_path='kitti_submission'):
    """ Create submission for the Sintel leaderboard """
    model.eval()
    test_dataset = datasets.KITTI(split='testing', aug_params=None)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for test_id in range(len(test_dataset)):
        image1, image2, (frame_id, ) = test_dataset[test_id]
        padder = InputPadder(image1.shape, mode='kitti')
        image1, image2 = padder.pad(image1[None].cuda(), image2[None].cuda())

        _, flow_pr = model(image1, image2, iters=iters, test_mode=True)
        flow = padder.unpad(flow_pr[0]).permute(1, 2, 0).cpu().numpy()

        output_filename = os.path.join(output_path, frame_id)
        frame_utils.writeFlowKITTI(output_filename, flow)


@torch.no_grad()
def validate_chairs(model, iters=24):
    """ Perform evaluation on the FlyingChairs (test) split """
    model.eval()
    epe_list = []

    val_dataset = datasets.FlyingChairs(split='validation')
    for val_id in range(len(val_dataset)):
        images, flow_gt, _ = val_dataset[val_id]
        images = images[None].cuda()

        _, flow_pr = model(images, iters=iters, test_mode=True)
        epe = torch.sum((flow_pr[0].cpu() - flow_gt)**2, dim=0).sqrt()
        epe_list.append(epe.view(-1).numpy())

    epe = np.mean(np.concatenate(epe_list))
    print("Validation Chairs EPE: %f" % epe)
    return {'chairs': epe}


@torch.no_grad()
def validate_sintel(model, iters=32, verbose=False, root='/result', output_dir='raw_flow'):
    """ Peform validation using the Sintel (train) split """
    model.eval()
    results = {}
    for dstype in ['final']:
        val_dataset = datasets.MpiSintel(split='training', dstype=dstype)
        epe_list = []

        for val_id in range(len(val_dataset)):
            images, flow_gt, _ = val_dataset[val_id]
            images = images[None].cuda()

            padder = InputPadder(images.shape)
            images = padder.pad(images)

            flow_low, flow_pr = model(images, iters=iters, test_mode=True)
            flow = padder.unpad(flow_pr[0]).cpu()

            epe = torch.sum((flow - flow_gt)**2, dim=0).sqrt()
            epe_list.append(epe.view(-1).numpy())

            if verbose:
                seq, frame_index = val_dataset.extra_info[val_id]
                output_dir_seq = f'{root}/{output_dir}/{seq}'
                if not os.path.exists(output_dir_seq):
                    os.makedirs(output_dir_seq)
                flow = flow.permute(1, 2, 0).numpy()
                # sintel_io.flow_write(f"{output_dir_seq}/frame_{frame_index:04d}.flo", flow)
                ofutil.writeFlowPng(f"{output_dir_seq}/frame_{frame_index:04d}.png", flow)

        epe_all = np.concatenate(epe_list)
        epe = np.mean(epe_all)
        px1 = np.mean(epe_all<1)
        px3 = np.mean(epe_all<3)
        px5 = np.mean(epe_all<5)

        print("Validation sintel (%s) EPE: %f, 1px: %f, 3px: %f, 5px: %f, >3px: %f" % (dstype, epe, px1, px3, px5, (1-px3)*100))
        results[dstype] = np.mean(epe_list)

    return results


@torch.no_grad()
def validate_kitti(model, iters=24, verbose=False, root='/result', output_dir='raw_flow'):
    """ Peform validation using the KITTI-2015 (train) split """
    model.eval()
    val_dataset = datasets.KITTI(split='training')

    out_list, epegt3_list, epe_list = [], [], []
    for val_id in range(len(val_dataset)):
        images, flow_gt, valid_gt = val_dataset[val_id]
        images = images[None].cuda()

        padder = InputPadder(images.shape, mode='kitti')
        images = padder.pad(images)

        flow_low, flow_pr = model(images, iters=iters, test_mode=True)
        flow = padder.unpad(flow_pr[0]).cpu()

        epe = torch.sum((flow - flow_gt)**2, dim=0).sqrt()
        mag = torch.sum(flow_gt**2, dim=0).sqrt()

        epe = epe.view(-1)
        mag = mag.view(-1)
        val = valid_gt.view(-1) >= 0.5

        epegt3 = ((epe > 3.0)).float()
        out = ((epe > 3.0) & ((epe/mag) > 0.05)).float()
        
        epe_list.append(epe[val].mean().item())
        epegt3_list.append(epegt3[val].cpu().numpy())
        out_list.append(out[val].cpu().numpy())

        if verbose:
            output_dir_seq = f'{root}/{output_dir}'
            if not os.path.exists(output_dir_seq):
                os.makedirs(output_dir_seq)
            flow = flow.permute(1, 2, 0).numpy()
            # sintel_io.flow_write(f"{output_dir_seq}/frame_{frame_index:04d}.flo", flow)
            ofutil.writeFlowPng(f"{output_dir_seq}/{val_id:06d}_10.png", flow)

    epe_list = np.array(epe_list)
    epegt3_list = np.concatenate(epegt3_list)
    out_list = np.concatenate(out_list)

    epe = np.mean(epe_list)
    epegt3 = 100 * np.mean(epegt3_list)
    f1 = 100 * np.mean(out_list)

    print("Validation KITTI AEPE:%f, EPE>3:%f, F1-all:%f" % (epe, epegt3, f1))
    return {'kitti-epe': epe, 'kitti-f1': f1}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--dataset', help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--stereo', action='store_true', help='use stereo mode')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    parser.add_argument('--use_fp16', action='store_true', help='use fp16 inference')
    parser.add_argument('--iters', type=int, default=6, help='number of iterations')
    parser.add_argument('--tradeoff', type=str, default='', help='use tradeoff model')
    parser.add_argument('--enable_quant', action='store_true')
    parser.add_argument('--enable_sparse', action='store_true')
    parser.add_argument('--verbose', action='store_true', help='verbose mode. enable it to dump raw flow maps')
    parser.add_argument('--outdir', type=str, default='raw_flow', help='output raw flow directory')
    args = parser.parse_args()
    
    if args.enable_quant:
        from pytorch_quantization import quant_modules
        quant_modules.initialize()
    model = torch.nn.DataParallel(RAFT(args))
    if args.enable_quant:
        quant_modules.deactivate()
    if args.enable_sparse:
        from apex.contrib.sparsity import ASP
        ASP.init_model_for_pruning(model)
        
    model.load_state_dict(torch.load(args.model))
    if args.use_fp16:
        model = model.half()
    model.cuda()
    model.eval()

    # create_sintel_submission(model.module, warm_start=True)
    # create_kitti_submission(model.module)

    with torch.no_grad():
        if args.dataset == 'chairs':
            validate_chairs(model.module, args.iters)

        elif args.dataset == 'sintel':
            validate_sintel(model.module, args.iters, args.verbose, output_dir=args.outdir)

        elif args.dataset == 'kitti':
            validate_kitti(model.module, args.iters, args.verbose, output_dir=args.outdir, root="./")
