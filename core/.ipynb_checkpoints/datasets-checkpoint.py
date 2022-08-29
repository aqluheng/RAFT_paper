# Data loading based on https://github.com/NVIDIA/flownet2-pytorch

import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F
from torch.utils.data.distributed import DistributedSampler

import os
import math
import random
from glob import glob
import os.path as osp
# import tensorflow as tf

from utils import frame_utils
from utils.augmentor import FlowAugmentor, SparseFlowAugmentor


# +
class FlowDataset(data.Dataset):
    def __init__(self, aug_params=None, sparse=False, preload=False, ret_extra_info=False, kitti_fmt=False, useAutoFlowAug=False):
        self.augmentor = None
        self.sparse = sparse
        if aug_params is not None:
            if sparse:
                self.augmentor = SparseFlowAugmentor(**aug_params)
            else:
                if useAutoFlowAug:
                    from utils.RAFT_augmentation import raftAugment                    
                    self.augmentor = raftAugment(**aug_params)
                else:
                    self.augmentor = FlowAugmentor(**aug_params)

        self.is_test = False
        self.init_seed = False
        self.preload = preload
        self.flow_list = []
        self.image_list = []
        self.extra_info = []
        self.ret_extra_info = ret_extra_info
        self.kitti_fmt = kitti_fmt
        self.useAutoFlowAug = useAutoFlowAug
        
    def __getitem__(self, index):
#         from time import time
#         lastTime = time()
        if self.is_test:
            img1 = frame_utils.read_gen(self.image_list[index][0])
            img2 = frame_utils.read_gen(self.image_list[index][1])
            img1 = np.array(img1).astype(np.uint8)[..., :3]
            img2 = np.array(img2).astype(np.uint8)[..., :3]
            img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
            img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
            return torch.cat((img1[None], img2[None]), dim=0), self.extra_info[index]

        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True

        index = index % len(self.image_list)
        valid = None
        if not self.preload:
            if self.sparse:
                # TODO: switch different flow reading functions in another way
                if 'VirtualKITTI2' in self.flow_list[index]:
                    flow, valid = frame_utils.read_vkitti_png_flow(self.flow_list[index])
                else:
                    flow, valid = frame_utils.readFlowKITTI(self.flow_list[index])
            else:
                if self.kitti_fmt:
                    flow, _ = frame_utils.readFlowKITTI(self.flow_list[index])
                else:
                    flow = frame_utils.read_gen(self.flow_list[index])

            img1 = frame_utils.read_gen(self.image_list[index][0])
            img2 = frame_utils.read_gen(self.image_list[index][1])
        else:
            flow, valid = self.flow_list[index]
            img1, ref_img = self.image_list[index]
            if isinstance(ref_img, int):
                img2 = self.image_list[index+1][0]
            else:
                img2 = ref_img
        flow = np.array(flow).astype(np.float32)
        img1 = np.array(img1).astype(np.uint8)
        img2 = np.array(img2).astype(np.uint8)

        # grayscale images
        if len(img1.shape) == 2:
            img1 = np.tile(img1[...,None], (1, 1, 3))
            img2 = np.tile(img2[...,None], (1, 1, 3))
        else:
            img1 = img1[..., :3]
            img2 = img2[..., :3]

        if self.augmentor is not None:
            if self.sparse:
                img1, img2, flow, valid = self.augmentor(img1, img2, flow, valid)
            else:
                if self.useAutoFlowAug:
#                     images =  tf.convert_to_tensor([img1.astype("float32"),img2.astype("float32")])/255.0
#                     flow = tf.convert_to_tensor(flow.astype("float32"))
                    images = np.stack((img1, img2)).astype("float32") / 255.0
                    flow = flow.astype("float32")
                    images, flow = self.augmentor(images, flow)
                    img1, img2 = images[0].numpy().astype("float32"), images[1].numpy().astype("float32")
                    flow = flow.numpy().astype("float32")
                    img1 = np.uint8((img1+1)*255)
                    img2 = np.uint8((img2+1)*255)
                else:
                    img1, img2, flow = self.augmentor(img1, img2, flow)
        
#         print(img1.min(),img1.max())
        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
        flow = torch.from_numpy(flow).permute(2, 0, 1).float()[:2]

        if valid is not None:
            valid = torch.from_numpy(valid)
        else:
            valid = (flow[0].abs() < 1000) & (flow[1].abs() < 1000)

        #img1 = img1[None,:,:,:]
        #img2 = img2[None,:,:,:]
#         print("time in getitem", time()-lastTime)

        if self.ret_extra_info:
            return torch.cat((img1[None], img2[None]), dim=0), flow, valid.float(), self.extra_info[index]
        else:
            return torch.cat((img1[None], img2[None]), dim=0), flow, valid.float()


    def __rmul__(self, v):
        assert(not self.preload)
        self.flow_list = v * self.flow_list
        self.image_list = v * self.image_list
        return self
        
    def __len__(self):
        return len(self.image_list)


# -

class MpiSintel(FlowDataset):
    def __init__(self, aug_params=None, split='training', root='/dataset/MPI-Sintel', dstype='clean', ret_extra_info=False):
        super(MpiSintel, self).__init__(aug_params, ret_extra_info=ret_extra_info)
        flow_root = osp.join(root, split, 'flow')
        image_root = osp.join(root, split, dstype)

        if split == 'test':
            self.is_test = True

        for scene in os.listdir(image_root):
            image_list = sorted(glob(osp.join(image_root, scene, '*.png')))
            for i in range(len(image_list)-1):
                self.image_list += [ [image_list[i], image_list[i+1]] ]
                self.extra_info += [ (scene, i) ] # scene and frame_id

            if split != 'test':
                self.flow_list += sorted(glob(osp.join(flow_root, scene, '*.flo')))


class FlyingChairs(FlowDataset):
    def __init__(self, aug_params=None, split='train', root='/dataset/FlyingChairs/FlyingChairs_release/data', useAutoFlowAug=False):
        super(FlyingChairs, self).__init__(aug_params, useAutoFlowAug=useAutoFlowAug)
        
        print("useAutoFlowAug:", useAutoFlowAug)
        images = sorted(glob(osp.join(root, '*.ppm')))
        flows = sorted(glob(osp.join(root, '*.flo')))
        assert (len(images)//2 == len(flows))

        split_list = np.loadtxt('chairs_split.txt', dtype=np.int32)
        for i in range(len(flows)):
            xid = split_list[i]
            if (split=='training' and xid==1) or (split=='validation' and xid==2):
                self.flow_list += [ flows[i] ]
                self.image_list += [ [images[2*i], images[2*i+1]] ]

class FlyingThingsSubset(FlowDataset):
    def __init__(self, aug_params=None, root='/dataset/FlyingThingsSubset'):
        super(FlyingThingsSubset, self).__init__(aug_params)

        for view in ['left', 'right']:
            image_root = osp.join(root, 'train/image_clean', view)
            flow_root_forward = osp.join(root, 'train/flow', view, 'into_future')
            flow_root_backward = osp.join(root, 'train/flow/', view, 'into_past')

            image_list = sorted(os.listdir(image_root))
            flow_forward = set(os.listdir(flow_root_forward))
            flow_backward = set(os.listdir(flow_root_backward))

            for i in range(len(image_list)-1):
                img1 = image_list[i]
                img2 = image_list[i+1]

                image_path1 = osp.join(image_root, img1)
                image_path2 = osp.join(image_root, img2)

                if img1.replace('.png', '.flo') in flow_forward:
                    self.image_list += [ [image_path1, image_path2] ]
                    self.flow_list += [ osp.join(flow_root_forward, img1.replace('.png', '.flo')) ]

                if img2.replace('.png', '.flo') in flow_backward:
                    self.image_list += [ [image_path2, image_path1] ]
                    self.flow_list += [ osp.join(flow_root_backward, img2.replace('.png', '.flo')) ]

class FlyingThings3D(FlowDataset):
    def __init__(self, aug_params=None, root='/dataset/FlyingThings3D_stereo/', dstype='frames_cleanpass'):
        super(FlyingThings3D, self).__init__(aug_params)
        root = osp.join(root, 'FlyingThings3D')

        for cam in ['left']:
            for direction in ['into_future', 'into_past']:
                image_dirs = sorted(glob(osp.join(root, dstype, 'TRAIN/*/*')))
                image_dirs = sorted([osp.join(f, cam) for f in image_dirs])

                flow_dirs = sorted(glob(osp.join(root, 'optical_flow/TRAIN/*/*')))
                flow_dirs = sorted([osp.join(f, direction, cam) for f in flow_dirs])

                for idir, fdir in zip(image_dirs, flow_dirs):
                    images = sorted(glob(osp.join(idir, '*.png')) )
                    flows = sorted(glob(osp.join(fdir, '*.pfm')) )
                    
                    for i in range(len(flows)-1):
                        assert flows[i].split('_')[-2] in os.path.basename(images[i]), "flow index mismatch with image"
                        if direction == 'into_future':
                            self.image_list += [ [images[i], images[i+1]] ]
                            self.flow_list += [ flows[i] ]
                        elif direction == 'into_past':
                            self.image_list += [ [images[i+1], images[i]] ]
                            self.flow_list += [ flows[i+1] ]

class Driving(FlowDataset):
    def __init__(self, aug_params=None, root='/dataset/FlyingThings3D_stereo/', dstype='frames_cleanpass'):
        super(Driving, self).__init__(aug_params)

        root = osp.join(root, 'Driving')

        for cam in ['left']:
            for direction in ['into_future', 'into_past']:
                image_dirs = sorted(glob(osp.join(root, dstype, '*/*/*')))
                image_dirs = sorted([osp.join(f, cam) for f in image_dirs])

                flow_dirs = sorted(glob(osp.join(root,  'optical_flow/*/*/*')))
                flow_dirs = sorted([osp.join(f, direction, cam) for f in flow_dirs])

                for idir, fdir in zip(image_dirs, flow_dirs):
                    images = sorted(glob(osp.join(idir, '*.png')) )
                    flows = sorted(glob(osp.join(fdir, '*.pfm')) )
                    
                    for i in range(len(flows)-1):
                        assert flows[i].split('_')[-2] in os.path.basename(images[i]), "flow index mismatch with image"
                        if direction == 'into_future':
                            self.image_list += [ [images[i], images[i+1]] ]
                            self.flow_list += [ flows[i] ]
                        elif direction == 'into_past':
                            self.image_list += [ [images[i+1], images[i]] ]
                            self.flow_list += [ flows[i+1] ]

# +
class Monkaa(FlowDataset):
    def __init__(self, aug_params=None, root='/dataset/FlyingThings3D_stereo/', dstype='frames_cleanpass'):
        super(Monkaa, self).__init__(aug_params)

        root = osp.join(root, 'Monkaa')

        for cam in ['left']:
            for direction in ['into_future', 'into_past']:
                image_dirs = sorted(glob(osp.join(root, dstype, '*')))
                image_dirs = sorted([osp.join(f, cam) for f in image_dirs])

                flow_dirs = sorted(glob(osp.join(root,  'optical_flow/*')))
                flow_dirs = sorted([osp.join(f, direction, cam) for f in flow_dirs])

                for idir, fdir in zip(image_dirs, flow_dirs):
                    images = sorted(glob(osp.join(idir, '*.png')) )
                    flows = sorted(glob(osp.join(fdir, '*.pfm')) )
                    
                    for i in range(len(flows)-1):
                        assert os.path.exists(images[i]) and os.path.exists(flows[i])
                        assert flows[i].split('_')[-2] in os.path.basename(images[i]), "flow index mismatch with image"
                        if direction == 'into_future':
                            self.image_list += [ [images[i], images[i+1]] ]
                            self.flow_list += [ flows[i] ]
                        elif direction == 'into_past':
                            self.image_list += [ [images[i+1], images[i]] ]
                            self.flow_list += [ flows[i+1] ]

class KITTI(FlowDataset):
    def __init__(self, aug_params=None, split='training', root='/dataset/Kitti2015/data_scene_flow', ret_extra_info=False):
        super(KITTI, self).__init__(aug_params, sparse=True, ret_extra_info=ret_extra_info)
        if split == 'testing':
            self.is_test = True

        root = osp.join(root, split)
        images1 = sorted(glob(osp.join(root, 'image_2/*_10.png')))
        images2 = sorted(glob(osp.join(root, 'image_2/*_11.png')))

        for img1, img2 in zip(images1, images2):
            frame_id = img1.split('/')[-1]
            self.extra_info += [ [frame_id] ]
            self.image_list += [ [img1, img2] ]

        if split == 'training':
            self.flow_list = sorted(glob(osp.join(root, 'flow_occ/*_10.png')))


# +
class moviSubset(FlowDataset):
    def __init__(self, aug_params=None, split='training', root='/raft_demo/MOVI/train',useAutoFlowAug=False):
        super(moviSubset,self).__init__(aug_params, sparse=False, kitti_fmt=True, useAutoFlowAug=useAutoFlowAug)
        if split == 'testing':
            self.is_test = True
        
#         root = osp.join(root, split)
        imgPath = osp.join(root, "images")
        forwardPath = osp.join(root, "forwardflow")
        
        videoList = []
        for video in os.listdir(imgPath):
            videoList.append(video)
        
        for video in videoList:
            videoImgPath = osp.join(imgPath, video)
            videoForwardPath = osp.join(forwardPath, video)
            
            for idx in range(23):
                self.image_list += [[osp.join(videoImgPath, "%02d.png"%idx), osp.join(videoImgPath, "%02d.png"%(idx+1))]]
                self.flow_list += [ osp.join(videoForwardPath, "%02d.png"%(idx)) ]

# +
class moviFilter(FlowDataset):
    def __init__(self, aug_params=None, split='training', root='/raft_demo/MOVI/train', filterPth='/raft_demo/RAFT_paper/filterMoviIndex.pth',useAutoFlowAug=False):
        super(moviFilter,self).__init__(aug_params, sparse=False, kitti_fmt=True, useAutoFlowAug=useAutoFlowAug)
        if split == 'testing':
            self.is_test = True
        
#         root = osp.join(root, split)
        imgPath = osp.join(root, "images")
        forwardPath = osp.join(root, "forwardflow")
        
        videoList = []
        for video in os.listdir(imgPath):
            videoList.append(video)
        
        All_image_list = []
        All_flow_list = []
        
        for video in videoList:
            videoImgPath = osp.join(imgPath, video)
            videoForwardPath = osp.join(forwardPath, video)
            
            for idx in range(23):
                All_image_list += [[osp.join(videoImgPath, "%02d.png"%idx), osp.join(videoImgPath, "%02d.png"%(idx+1))]]
                All_flow_list += [ osp.join(videoForwardPath, "%02d.png"%(idx)) ]
        
        import torch
        filterIdx = torch.load(filterPth)
        for idx in filterIdx:
            self.image_list += [All_image_list[idx]]
            self.flow_list += [All_flow_list[idx]]


# -

class HD1K(FlowDataset):
    def __init__(self, aug_params=None, preload=False, root='/dataset/HD1K'):
        super(HD1K, self).__init__(aug_params, sparse=True, preload=preload)

        seq_ix = 0
        while 1:
            flows = sorted(glob(os.path.join(root, 'hd1k_flow_gt', 'flow_occ/%06d_*.png' % seq_ix)))
            images = sorted(glob(os.path.join(root, 'hd1k_input', 'image_2/%06d_*.png' % seq_ix)))

            if len(flows) == 0:
                break

            for i in range(len(flows)-1):
                if not self.preload:
                    self.flow_list += [flows[i]]
                    self.image_list += [ [images[i], images[i+1]] ]
                else:           
                    self.flow_list += [frame_utils.readFlowKITTI(flows[i])]
                    ref_img = None
                    if i == len(flows)-2:
                        ref_img = i+1
                    else:
                        ref_img = frame_utils.read_gen(images[i+1])
                    self.image_list += [ [frame_utils.read_gen(images[i]), ref_img] ]
            
            seq_ix += 1

class TUM(FlowDataset):
    def __init__(self, aug_params=None, root='/dlof/TUM'):
        super(TUM, self).__init__(aug_params, sparse=True)
        seq_ix = 1
        while 1:
            flows = sorted(glob(os.path.join(root, 'png_flow/%02d_*.png' % seq_ix)))
            images = sorted(glob(os.path.join(root, 'cam/%02d_*.png' % seq_ix)))

            if len(flows) == 0:
                break

            for i in range(len(flows)-1):
                self.flow_list += [flows[i]]
                self.image_list += [ [images[i], images[i+1]] ]
            
            seq_ix += 1

class AutoFlow(FlowDataset):
    def __init__(self, aug_params=None, root='/dataset/Autoflow', useAutoFlowAug=False):
        super(AutoFlow, self).__init__(aug_params, sparse=False, useAutoFlowAug=useAutoFlowAug)
        for path in os.listdir(root):
            self.flow_list += [osp.join(root,path,"forward.flo")]
            self.image_list += [ [osp.join(root,path,"im0.png"), osp.join(root,path,"im1.png") ] ]


class VirtualKITTI2(FlowDataset):
    def __init__(self, aug_params=None, root='/dataset/VirtualKITTI2'):
        super(VirtualKITTI2, self).__init__(aug_params, sparse=True)
        for cam in ['Camera_0']:
            sub_dirs = sorted(glob(f'{root}/*/*'))
            for idir in sub_dirs:
                images = sorted(glob(f'{idir}/frames/rgb/{cam}/*.jpg'))
                for direction in ['forwardFlow', 'backwardFlow']:
                    flows = sorted(glob(f'{idir}/frames/{direction}/{cam}/*.png'))
                    if direction == 'forwardFlow':
                        self.image_list += list(zip(images[:-1], images[1:]))
                    elif direction == 'backwardFlow':
                        self.image_list += list(zip(images[1:], images[:-1]))
                    self.flow_list += flows

def fetch_dataloader(args, TRAIN_DS=''):
    """ Create the data loader for the corresponding trainign set """

    if args.stage == 'chairs':
        if args.aug_setting == "kittiAug":
            aug_params = {'crop_size': (368, 496), 'min_scale': -0.1, 'max_scale': 1.0, 'do_flip': True}
            train_dataset = FlyingChairs(aug_params, split='training')
        elif args.aug_setting == "autoflowAug":
            aug_params = {'crop_size': (320, 448), 'min_scale': -0.2, 'max_scale': 0.5}
            train_dataset = FlyingChairs(aug_params, split='training', useAutoFlowAug=True)
        
    
    elif args.stage == 'things':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.4, 'max_scale': 0.8, 'do_flip': True}
        clean_dataset = FlyingThings3D(aug_params, dstype='frames_cleanpass')
        # clean_dataset += Driving(aug_params, dstype='frames_cleanpass')
        # clean_dataset += Monkaa(aug_params, dstype='frames_cleanpass')

        final_dataset = FlyingThings3D(aug_params, dstype='frames_finalpass')
        # final_dataset += Driving(aug_params, dstype='frames_finalpass')
        # final_dataset += Monkaa(aug_params, dstype='frames_finalpass')

        train_dataset = clean_dataset + final_dataset

    elif args.stage == 'thingsSubset':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.4, 'max_scale': 0.8, 'do_flip': True}
        train_dataset = FlyingThingsSubset(aug_params)

    elif args.stage == 'sintel':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.2, 'max_scale': 0.6, 'do_flip': True}
        sintel_clean = MpiSintel(aug_params, split='training', dstype='clean')
        sintel_final = MpiSintel(aug_params, split='training', dstype='final')        

        if TRAIN_DS == 'C+T+K+S+H':
            kitti = KITTI({'crop_size': args.image_size, 'min_scale': -0.3, 'max_scale': 0.5, 'do_flip': True})
            things = FlyingThings3D(aug_params, dstype='frames_cleanpass')
            hd1k = HD1K({'crop_size': args.image_size, 'min_scale': -0.5, 'max_scale': 0.2, 'do_flip': True})
            train_dataset = 100*sintel_clean + 100*sintel_final + 200*kitti + 5*hd1k + things

        elif TRAIN_DS == 'C+T+K/S':
            things = FlyingThings3D(aug_params, dstype='frames_cleanpass')
            train_dataset = 100*sintel_clean + 100*sintel_final + things

        elif TRAIN_DS == '':
            train_dataset = sintel_clean + sintel_final

    elif args.stage == 'kitti':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.2, 'max_scale': 0.4, 'do_flip': False}
        train_dataset = KITTI(aug_params, split='training')

    elif args.stage == 'hd1k':
        hd1k = HD1K({'crop_size': args.image_size, 'min_scale': -0.5, 'max_scale': 0.2, 'do_flip': True})
        if TRAIN_DS == 'T+S+H':
            aug_params = {'crop_size': args.image_size, 'min_scale': -0.2, 'max_scale': 0.6, 'do_flip': True}
            things = FlyingThings3D(aug_params, dstype='frames_cleanpass')
            sintel_clean = MpiSintel(aug_params, split='training', dstype='clean')
            train_dataset = 100*sintel_clean + 5*hd1k + things
        elif TRAIN_DS == '':
            train_dataset = hd1k

    elif args.stage == 'TUM':
        tum = TUM({'crop_size': args.image_size, 'min_scale': -0.2, 'max_scale': 0.4, 'do_flip': True})
        if TRAIN_DS == 'TUM+H':
            hd1k = HD1K({'crop_size': args.image_size, 'min_scale': -0.5, 'max_scale': 0.2, 'do_flip': True})
            train_dataset = tum+hd1k
        else:
            train_dataset = tum
        
    elif args.stage == 'virtualkitti2':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.2, 'max_scale': 0.4, 'do_flip': False}
        vkitti = VirtualKITTI2(aug_params)
        if TRAIN_DS == 'T+V':
            things = FlyingThings3D(aug_params, dstype='frames_cleanpass')
            train_dataset = vkitti + things
        elif TRAIN_DS == '':
            train_dataset = vkitti
    elif args.stage == "MoviFilter":
        if args.aug_setting == "kittiAug":
            aug_params = {'crop_size': (256, 256), 'min_scale': -0.2, 'max_scale': 0.4, 'do_flip': False}
            train_dataset = moviFilter(aug_params)
        elif args.aug_setting == "autoflowAug":
            aug_params = {'crop_size': (256, 256), 'min_scale': -0.2, 'max_scale': 0.5}
            train_dataset = moviFilter(aug_params, useAutoFlowAug=True)
    elif args.stage.startswith("movi"):
        if args.aug_setting == "kittiAug":
            aug_params = {'crop_size': (256, 256), 'min_scale': -0.2, 'max_scale': 0.4, 'do_flip': False}
            train_dataset = moviSubset(aug_params)
        elif args.aug_setting == "autoflowAug":
            aug_params = {'crop_size': (320, 448), 'min_scale': -0.2, 'max_scale': 0.5}
            train_dataset = moviSubset(aug_params, useAutoFlowAug=True)
    elif args.stage.startswith("autoflow"):
        if args.aug_setting == "kittiAug":
            aug_params = {'crop_size': (224, 288), 'min_scale': -0.2, 'max_scale': 0.4, 'do_flip': False}
            train_dataset = AutoFlow(aug_params)
        elif args.aug_setting == "autoflowAug":
            aug_params = {'crop_size': (320, 448), 'min_scale': -0.2, 'max_scale': 0.5}
            train_dataset = AutoFlow(aug_params, useAutoFlowAug=True)
            
    train_sampler = DistributedSampler(train_dataset, shuffle=True) if args.dist else None
    train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, pin_memory=True, 
        sampler=train_sampler, shuffle=(train_sampler is None), num_workers=args.num_workers, drop_last=True,
        persistent_workers=True)

    print('Training with %d image pairs' % len(train_dataset))
    return train_loader


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from matplotlib import gridspec
    import cv2
    import numpy as np
    def drawConcat(images, flow):
        def renderFlowImg(flow, *, maxFlow=-1, style='mpi'):
            MVX_PLANE   = 0
            MVY_PLANE   = 1
            assert style in ['kitti-c++', 'mpi'], 'unknown flow rendering style: {}'.format(style)
            assert len(flow.shape) == 3
            mvx = flow[:,:,MVX_PLANE]
            mvy = flow[:,:,MVY_PLANE]
            magnitude = np.sqrt(mvx ** 2 + mvy ** 2)
            direction = np.arctan2(mvy, mvx)
            if maxFlow < 0:
                if style == 'kitti-c++':
                    maxFlow = np.fmax(magnitude.max(), 1.0)
                else:
                    # mpi style
                    maxFlow = 1.2 * max(np.fabs(mvx).max(), np.fabs(mvy).max())
                    maxFlow = np.fmax(maxFlow, 1.0)
                print('maxFlow = {:.1f}'.format(maxFlow))
            h = np.fmod(direction / (2 * np.pi) + 1.0, 1.0) * 360
            assert h.min() >= 0
            if style == 'kitti-c++':
                """KITTI C++ hsv2rgb has bug. Here tries to mimic the final result"""
                s = np.clip(magnitude * 300 / maxFlow, 0, 1.0)
                v = np.clip(magnitude * 8   / maxFlow, 0, 1.0)
            else:
                # mpi style
                n = 8
                s = np.clip(magnitude * n / maxFlow, 0, 1.0)
                v = np.clip(n - s, 0, 1.0)
            img = cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2RGB)
        #     img[~valid] = 0
        #     return np.uint8(img * 255)[:,:,[1,2,0]]
            return np.uint8(img * 255)

        nrow, ncol = 1, 3
        fig = plt.figure(figsize=(ncol*5, nrow*5))

        gs = gridspec.GridSpec(nrow, ncol,
             wspace=0.0, hspace=0.0, 
             top=(1.-0.5/(nrow+1))*5, bottom=5*(0.5/(nrow+1)), 
             left=(0.5/(ncol+1)), right=(1-0.5/(ncol+1))) 

        ax0 = plt.subplot(gs[0,0])
        ax1 = plt.subplot(gs[0,1])
        ax2 = plt.subplot(gs[0,2])

        ax0.imshow(images[0])
        ax0.axis("off")
        ax1.imshow(images[1])
        ax1.axis("off")
        ax2.imshow(renderFlowImg(flow))
        ax2.axis("off")
        plt.show()


if __name__ == '__main__':
    train_dataset = AutoFlow()

if __name__ == '__main__':
#     movi_dataset = moviSubset()
#     print(len(movi_dataset))
    import argparse
    import datetime
    import torch.distributed as dist
    args = argparse.Namespace()
    args.stage = "movi"
    args.aug_setting = "autoflowAug"
    args.dist = False
    args.batch_size = 1
    args.num_workers = 7
    args.local_rank = 0
    if args.dist:
        dist.init_process_group(backend='nccl', init_method='env://',timeout=datetime.timedelta(seconds=7200))
        device = torch.device(f'cuda:{args.local_rank}')
        args.device = device
        torch.cuda.set_device(args.local_rank)
    args.world_size = dist.get_world_size() if args.dist else 1
    loader = fetch_dataloader(args)

if __name__ == '__main__':
#     movi_dataset = moviSubset()
#     print(len(movi_dataset))
    import argparse
    import datetime
    import torch.distributed as dist
    args = argparse.Namespace()
    args.dist = False
    args.stage = "autoflow"
    args.aug_setting = "kittiAug"
    args.batch_size = 1
    args.num_workers = 7
    args.local_rank = 0
    loader = fetch_dataloader(args)

if __name__ == '__main__':
    for images, flow, _ in loader:
        drawFlow = np.float32(flow[0].permute(1,2,0))
        drawConcat(np.uint8(images[0].permute(0,2,3,1)), drawFlow)
        break

if __name__ == '__main__':
    sintel_clean = MpiSintel(None, split='training', dstype='clean')
    sintel_final = MpiSintel(None, split='training', dstype='final')
    train_dataset = sintel_clean + sintel_final
    train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, pin_memory=True, 
        sampler=None, shuffle=True, num_workers=args.num_workers, drop_last=True,
        persistent_workers=True)
    for images, flow, _ in train_loader:
        drawFlow = np.float32(flow[0].permute(1,2,0))
        drawConcat(np.uint8(images[0].permute(0,2,3,1)), drawFlow)
        break
