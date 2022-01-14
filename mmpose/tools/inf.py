import sys

from torch.utils import data
[sys.path.append(i) for i in ['.', '..']]

import argparse
import os
import os.path as osp

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.runner import init_dist

from mmpose.apis import init_pose_model, inference_top_down_pose_model, vis_pose_result
from PIL import Image

import numpy as np
import pandas as pd
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description='mmpose inference model')
    parser.add_argument('config', help='inference config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--out', help='output result file')
    parser.add_argument(
        '--work-dir', help='the dir to save evaluation results')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument(
        '--eval',
        default=None,
        nargs='+',
        help='evaluation metric, which depends on the dataset,'
        ' e.g., "mAP" for MSCOCO')
    parser.add_argument(
        '--gpu_collect',
        action='store_true',
        help='whether to use gpu to collect results')
    parser.add_argument('--tmpdir', help='tmp dir for writing some results')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        default={},
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. For example, '
        "'--cfg-options model.backbone.depth=18 model.backbone.with_cp=True'")
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def merge_configs(cfg1, cfg2):
    # Merge cfg2 into cfg1
    # Overwrite cfg1 if repeated, ignore if value is None.
    cfg1 = {} if cfg1 is None else cfg1.copy()
    cfg2 = {} if cfg2 is None else cfg2
    for k, v in cfg2.items():
        if v:
            cfg1[k] = v
    return cfg1


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True
    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])
    
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    
    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
        
    model = init_pose_model(cfg, checkpoint=args.checkpoint, device='cuda:0')
    upper_dir = '/nas1/yjun/slt/PHOENIX-2014-T/features/fullFrame-210x260px/dev/'
    lower_dir = sorted(os.listdir(upper_dir))
    
    keypoints_list = pd.read_csv('/nas1/yjun/slt/keypoints.csv')['keypoint']
    keypoints_list_xy = [a + b for a in keypoints_list for b in ['_x', '_y']]
    for dir in tqdm(lower_dir, desc="directory"):
        img_list = sorted(os.listdir(upper_dir + dir))
        saving_results = np.array([]).reshape(-1,266)
        for img in tqdm(img_list, desc="image"):
            img_data = Image.open(upper_dir + dir + '/' + img)
            imgArray = np.asarray(img_data)
            results = inference_top_down_pose_model(model, 
                                                imgArray
                                                )
            saving_results = np.vstack((saving_results, results[0][0]['keypoints'][:,:2].reshape(1,266)))
            
            # visualization
            # data = vis_pose_result(model,
            #                     imgArray,
            #                     results[0],
            #                     kpt_score_thr=0.1)
            
            # img_keypoint = Image.fromarray(data, 'RGB')
            # img_keypoint.save('/nas1/yjun/slt/keypoint/' + img + '_keypoint.png')
        
        df = pd.DataFrame(saving_results, index = img_list, columns=keypoints_list_xy)
        df.to_csv(f'/nas1/yjun/slt/{dir}.csv')
    

if __name__ == '__main__':
    main()
