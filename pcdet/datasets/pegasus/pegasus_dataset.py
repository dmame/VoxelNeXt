import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
import pickle
import json
import random
import operator
from numba.cuda.simulator.api import detect
import numpy as np

from functools import reduce
from pathlib import Path
from copy import deepcopy

import copy
import pickle
from pathlib import Path

import numpy as np
from tqdm import tqdm

from ...ops.roiaware_pool3d import roiaware_pool3d_utils
from ...utils import common_utils
from ..dataset import DatasetTemplate
from .pegasus_utils import LoadPointCloudFromFile, LoadPointCloudAnnotations



class Pegasus3D_OD_Dataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
        root_path = (root_path if root_path is not None else Path(dataset_cfg.DATA_PATH)) 
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.load_interval = 1
        self.sequential_bev = False
        self.use_sensor_info = dict()
        self.sample = False
        self.nsweeps = 1
        self.class_table = None
        self.NumPointFeatures = 5
        self.test_mode=True
        self.key_frame_index_in_multi_sweeps = 5

        self.load_infos(self.mode)

        self.load_points = LoadPointCloudFromFile()
        self.load_annotations = LoadPointCloudAnnotations()

    def load_infos(self, mode):
        self.logger.info('Loading pegasus dataset')

        # info_path = self._info_path if isinstance(self._info_path, (tuple,list)) else [self._info_path]

        _pegasus_infos_all = dict()
        for info_path in self.dataset_cfg.INFO_PATH[mode]:
            info_path = self.root_path / info_path
            with open(info_path, "rb") as f:
                single_pegasus_infos = pickle.load(f)
                assert isinstance(single_pegasus_infos, dict)
                _pegasus_infos_all.update(single_pegasus_infos)
        self._pegasus_infos_all = _pegasus_infos_all

        ### convert dict to list for getitem
        pegasus_key_frame_infos = []
        for sample_id, sample_info in tqdm(_pegasus_infos_all.items()):
            if sample_info['bbox_category'] is not None and sample_info['bbox'] is not None:
                if sample_info['path']['pcd'].endswith('/'):
                    sample_info['path']['pcd'] = sample_info['path']['pcd'][:-1]
                if not os.path.exists(sample_info['path']['pcd']):
                    continue
                sample_pose_info = self.get_pose_info(sample_info)
                if sample_pose_info is None:
                    continue
                sample_info['bbox_category'] = self.map_category(sample_info['bbox_category'])
                single_clp_id_info = dict()
                single_clp_id_info[sample_id] = sample_info
                pegasus_key_frame_infos.append(single_clp_id_info)
        
        # ### use the load_interval to sample data
        self._pegasus_key_frame_infos = pegasus_key_frame_infos[::self.load_interval]
            
        self.logger.info("Using {} Key Frames from total {} frames".format(len(self._pegasus_key_frame_infos), len(_pegasus_infos_all.items())))

    def get_pose_info(self, sample_info):
        if sample_info['motion_compensated']:
            pose_info_index = 5
        else:
            pose_info_index = 0
            
        if sample_info['pose_info'] is None or sample_info['pose_info'][pose_info_index] is None:
            return None
        
        return sample_info['pose_info'][pose_info_index]
    
    def map_category(self, category_list):
        """mapping classes"""
        if self.class_table is not None:
            new_cat_list = [self.class_table [cat] for cat in category_list]
            return new_cat_list
        else:
            return category_list


    def __len__(self):
        if self._merge_all_iters_to_one_epoch:
            return len(self._pegasus_key_frame_infos) * self.total_epochs

        return len(self._pegasus_key_frame_infos)

    def get_sweep_info(self, sample_id):
        sweep_info = dict()
        sweep_clp_info = self._pegasus_infos_all[sample_id]
        sweep_clp_pose_info = self.get_pose_info(sweep_clp_info)
        if sweep_clp_pose_info is None:
            return None
        
        if sweep_clp_info['path']['pcd'].endswith('/'):
            sweep_clp_info['path']['pcd'] = sweep_clp_info['path']['pcd'][:-1]
        
        if sweep_clp_info['path']['pcd'] is None or not os.path.exists(sweep_clp_info['path']['pcd']):
            return None
        
        sweep_info[sample_id] = sweep_clp_info
        return sweep_info
    
    def get_sensor_data(self, idx):
        info = self._pegasus_key_frame_infos[idx]
        assert isinstance(info, dict), "Pegasus Dataset v2 is Dict"
        assert len(info.keys()) == 1
        sample_id = list(info.keys())[0]
        sample_info = list(info.values())[0]
        
        if 'token' not in info:
            sample_info['token'] = sample_info['timestamp']

        res = {
            "sample_id": sample_id,
            'sample_info': sample_info,
            "nsweeps": self.nsweeps,
            "sweeps_info": None,
            'use_sensor_info': self.use_sensor_info,
            "num_point_features": self.NumPointFeatures,
            "mode": "val" if self.test_mode else "train",
            "type": "PegasusDataset",
            'lidar':dict(),
            'metadata': {
                'token':  sample_id,
            }
        }
        
        if self.nsweeps > 1:
            sweeps_info = []
            prev_id = 0
            while len(sweeps_info) < self.nsweeps - 1:
                prev_id += 1
                if prev_id > self.key_frame_index_in_multi_sweeps:
                    if len(sweeps_info):
                        sweeps_info.append(deepcopy(sweeps_info[-1]))
                    else:
                        tmp_sample_info = deepcopy(info)
                        tmp_sample_info[sample_id]['sweep_interval_num'] = 0
                        sweeps_info.append(tmp_sample_info)
                else:       
                    adj_prev_sweep_sample_id = sample_info['multi_sweep'][self.key_frame_index_in_multi_sweeps - prev_id]  ## key frame index is 5
                    if adj_prev_sweep_sample_id is None:
                        tmp_sample_info = deepcopy(info)
                        tmp_sample_info[sample_id]['sweep_interval_num'] = 0
                        sweeps_info.append(tmp_sample_info)
                        continue
                    
                    adj_prev_sweep_clp_info = self.get_sweep_info(adj_prev_sweep_sample_id)
                    if adj_prev_sweep_clp_info is None:
                        tmp_sample_info = deepcopy(info)
                        tmp_sample_info[sample_id]['sweep_interval_num'] = 0
                        sweeps_info.append(tmp_sample_info)
                        continue
                     
                    adj_prev_sweep_clp_info[adj_prev_sweep_sample_id]['sweep_interval_num'] = prev_id
                    sweeps_info.append(adj_prev_sweep_clp_info)
            res['sweeps_info'] = sweeps_info

        return res

    def __getitem__(self, index):
        data_dict = self.get_sensor_data(index)
        print("2")
        # self.load_points(data_dict， None)
        # self.load_annotations
        

        print("1")
        # info = copy.deepcopy(self.infos[index])
        # points = self.get_lidar_with_sweeps(index, max_sweeps=self.dataset_cfg.MAX_SWEEPS)

        # input_dict = {
        #     'points': points,
        #     'frame_id': Path(info['lidar_path']).stem,
        #     'metadata': {'token': info['token']}
        # }

        # if 'gt_boxes' in info:
        #     if self.dataset_cfg.get('FILTER_MIN_POINTS_IN_GT', False):
        #         mask = (info['num_lidar_pts'] > self.dataset_cfg.FILTER_MIN_POINTS_IN_GT - 1)
        #     else:
        #         mask = None

        #     input_dict.update({
        #         'gt_names': info['gt_names'] if mask is None else info['gt_names'][mask],
        #         'gt_boxes': info['gt_boxes'] if mask is None else info['gt_boxes'][mask]
        #     })

        # data_dict = self.prepare_data(data_dict=input_dict)

        # if self.dataset_cfg.get('SET_NAN_VELOCITY_TO_ZEROS', False) and 'gt_boxes' in info:
        #     gt_boxes = data_dict['gt_boxes']
        #     gt_boxes[np.isnan(gt_boxes)] = 0
        #     data_dict['gt_boxes'] = gt_boxes

        # if not self.dataset_cfg.PRED_VELOCITY and 'gt_boxes' in data_dict:
        #     data_dict['gt_boxes'] = data_dict['gt_boxes'][:, [0, 1, 2, 3, 4, 5, 6, -1]]

        return data_dict


