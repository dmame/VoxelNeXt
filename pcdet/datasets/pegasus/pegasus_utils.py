import os
import copy
import pickle

import numpy as np
from pypcd import pypcd
from pyquaternion import Quaternion


class LoadPointCloudFromFile_Pegasus(object):
    def __init__(self, 
                 time_lag_frep=0.1, 
                 root_path=None, 
                 nsweeps=1,
                 use_dim=5, 
                 rm_points_radius=1, 
                 dummy=False,
                 **kwargs):
        self.random_select = kwargs.get("random_select", False)
        self.npoints = kwargs.get("npoints", 140000)
        self.time_lag_frep = time_lag_frep
        self.root_path = root_path
        self.nsweeps = nsweeps
        self.use_dim = use_dim
        self.rm_points_radius = rm_points_radius
        self.dummy = dummy
        # if nsweeps == 1:
        #     assert use_dim == 4, "nsweeps == 1, then use_dim should set to 4"
    
    def get_lidar2gloabl_transfromation(self, sample_info):
        sample_lidar2ego = np.eye(4)
        sample_lidar2ego_quat_pose = sample_info['lidar2ego']
        lidar2ego_qx, lidar2ego_qy, lidar2ego_qz, lidar2ego_qw, lidar2ego_x, lidar2ego_y, lidar2ego_z = sample_lidar2ego_quat_pose
        sample_lidar2ego_rot = np.array(Quaternion(lidar2ego_qw, lidar2ego_qx, lidar2ego_qy, lidar2ego_qz).rotation_matrix)
        sample_lidar2ego_tran = np.array([lidar2ego_x, lidar2ego_y, lidar2ego_z])
        sample_lidar2ego[:3, :3] = sample_lidar2ego_rot
        sample_lidar2ego[:3, -1] = sample_lidar2ego_tran
        
        if sample_info['motion_compensated']:
            pose_info_index = 5
        else:
            pose_info_index = 0
        
        sample_ego2gloabl = np.eye(4)
        sample_ego2gloabl_quat_pose = sample_info['pose_info'][pose_info_index]
        _, ego2global_qx, ego2global_qy, ego2global_qz, ego2global_qw, ego2global_x, ego2global_y , ego2global_z, vx, vy, vz = sample_ego2gloabl_quat_pose
        sample_ego2gloabl_rot = np.array(Quaternion(ego2global_qw, ego2global_qx, ego2global_qy, ego2global_qz).rotation_matrix)
        sample_ego2gloal_tran = np.array([ego2global_x, ego2global_y, ego2global_z])
        sample_ego2gloabl[:3, :3] = sample_ego2gloabl_rot
        sample_ego2gloabl[:3, -1] = sample_ego2gloal_tran
        
        sample_lidar2global = sample_ego2gloabl @ sample_lidar2ego
        
        return sample_lidar2global
    
    @staticmethod
    def load_pcd(filename):
        # assert os.path.exists(filename), f'{filename} not exists'
        pc = pypcd.PointCloud.from_path(filename)
        np_x = (np.array(pc.pc_data['x'], dtype=np.float32)).astype(np.float32)
        np_y = (np.array(pc.pc_data['y'], dtype=np.float32)).astype(np.float32)
        np_z = (np.array(pc.pc_data['z'], dtype=np.float32)).astype(np.float32)
        np_i = (np.array(pc.pc_data['intensity'], dtype=np.float32)).astype(np.float32)
        
        if len(pc.pc_data[0]) < 5:
            np_t = np.zeros_like(np_x)
        else:
            np_t = (np.array(pc.pc_data['timestamp'], dtype=np.float32)).astype(np.float32)
        
        return np.stack([np_x, np_y, np_z, np_i, np_t], axis=-1)
    
    def load_points(self, path, use_dim=5):
        # points = np.fromfile(path, dtype=np.float32) ### (x,y,z,i,t)
        points = self.load_pcd(path)
        points[:, 3] = points[:, 3] / 255.0  ## normalize intensity
        self._remove_close(points, radius=self.rm_points_radius)
        return points[:, :use_dim]
    
    def load_sweep_points(self, path, sweep_sample_lidar2cur_sample_lidar=None, use_dim=5):
        sweep_points = self.load_points(path, use_dim=use_dim)  ### shape:(N, use_dim)
        sweep_points_T = sweep_points.T  ### shape(use_dim, N)
        if sweep_sample_lidar2cur_sample_lidar is not None:
            sweep_points_T[:3, :] = sweep_sample_lidar2cur_sample_lidar.dot( 
                np.vstack((sweep_points_T[:3, :], np.ones_like(sweep_points_T[0:1, :])))
            )[:3, :]
        
        return sweep_points_T.T  ## return shape (N, use_dim)
    
    def _remove_close(self, points, radius=1.0):
        """Removes point too close within a certain radius from origin.

        Args:
            points (np.ndarray | :obj:`BasePoints`): Sweep points.
            radius (float, optional): Radius below which points are removed.
                Defaults to 1.0.

        Returns:
            np.ndarray: Points after removing.
        """
        if isinstance(points, np.ndarray):
            points_numpy = points
            
        x_filt = np.abs(points_numpy[:, 0]) < radius
        y_filt = np.abs(points_numpy[:, 1]) < radius
        not_close = np.logical_not(np.logical_and(x_filt, y_filt))
        return points[not_close]
    
    def __call__(self, res, info):
        if self.dummy:
            dummy_points = np.zeros((1, self.use_dim), dtype=np.float32)
            res['lidar']['points'] = dummy_points
            res['lidar']['cur_point_mask'] = np.zeros(1, dtype=np.bool_)
            return res, info

        if self.root_path is not None:
            path = os.path.join(self.root_path, res['sample_info']['path']['pcd'])
        else:
            path = res['sample_info']['path']['pcd']
        
        nsweeps = res["nsweeps"]
        assert nsweeps == self.nsweeps
        
        cur_sample_lidar_points = self.load_points(path, use_dim=self.use_dim)
        res['lidar']['points'] = copy.deepcopy(cur_sample_lidar_points) ### cur points
        
        if nsweeps > 1: 
            cur_timestamp = float(res['sample_info']['timestamp']) / 1e6
            sweep_points_list = [cur_sample_lidar_points[:, :4]]
            sweep_times_list = [np.zeros((cur_sample_lidar_points.shape[0], 1))]
            cur_point_mask = np.ones(cur_sample_lidar_points.shape[0], dtype=np.bool_)
            sweep_points_mask_list = [cur_point_mask]

            assert (nsweeps - 1) == len(res["sweeps_info"]), "nsweeps {} should be equal to the list length {}.".format(
                nsweeps, len(res["sweeps_info"])
            )
            
            if res.get('cur_sample_lidar2gloabl', None):
               cur_sample_lidar2gloabl = res['cur_sample_lidar2gloabl']
            else:
               cur_sample_lidar2gloabl = self.get_lidar2gloabl_transfromation(res['sample_info']) 
            
            sweep_samples_lidar2cur_sample_lidar = res.get('sweep_samples_lidar2cur_sample_lidar', None)
            sweep_samples_lidar2cur_sample_lidar_list = []
            for i in range(nsweeps - 1):
                sweep_info_dict = res["sweeps_info"][i]
                sweep_info = list(sweep_info_dict.values())[0]
                sweep_timestamp = float(sweep_info['timestamp']) / 1e6
                if self.root_path is not None:  ### use ral path
                    sweep_path = os.path.join(self.root_path, sweep_info['path']['pcd'])
                else:  ### use abs path
                    sweep_path = sweep_info['path']['pcd']
                    
                if sweep_samples_lidar2cur_sample_lidar is None:
                    sweep_sample_lidar2global = self.get_lidar2gloabl_transfromation(sweep_info)
                    sweep_sample_lidar2cur_sample_lidar = np.linalg.inv(cur_sample_lidar2gloabl) @ sweep_sample_lidar2global
                    sweep_samples_lidar2cur_sample_lidar_list.append(sweep_sample_lidar2cur_sample_lidar)
                else:
                    sweep_sample_lidar2cur_sample_lidar = sweep_samples_lidar2cur_sample_lidar[i]    
                
                points_sweep = self.load_sweep_points(sweep_path, sweep_sample_lidar2cur_sample_lidar, use_dim=self.use_dim)
                if self.time_lag_frep is None:
                    times_sweep = np.ones_like(points_sweep[:, 0:1]) * (cur_timestamp - sweep_timestamp)
                else:
                    times_sweep = np.ones_like(points_sweep[:, 0:1]) * self.time_lag_frep * sweep_info['sweep_interval_num']
                
                sweep_points_list.append(points_sweep[:, :4])
                sweep_times_list.append(times_sweep)
                sweep_points_mask_list.append(np.zeros(points_sweep.shape[0], dtype=np.bool_))

            points = np.concatenate(sweep_points_list, axis=0)
            times = np.concatenate(sweep_times_list, axis=0).astype(points.dtype)
            cur_point_mask = np.concatenate(sweep_points_mask_list, axis=0)

            res['lidar']["combined"] = np.hstack([points, times])  ## sweeep points combine
            res['lidar']['cur_point_mask'] = cur_point_mask
            
            if sweep_samples_lidar2cur_sample_lidar is None:
                res['sweep_samples_lidar2cur_sample_lidar'] = sweep_samples_lidar2cur_sample_lidar_list

        return res, info


class LoadAnnotations_Pegasus(object):
    def __init__(self, with_bbox=True, with_vel=True, **kwargs):
        self.with_vel = with_vel

    def __call__(self, res, info):
        
        gt_bboxes_3d = np.array(res['sample_info']['bbox'])
        if self.with_vel:  ## (x, y ,z ,w, l , h, vx ,vy ,theta)
            gt_bboxes_3d = gt_bboxes_3d[:, [0, 1, 2, 4, 3, 5, 6, 7, 10]]
        else:  ## (x, y ,z ,w, l , h, theta)
            gt_bboxes_3d = gt_bboxes_3d[:, [0, 1, 2, 4, 3, 5, 10]]
        
        #### convert theta to second coord theta
        gt_bboxes_3d[:, -1] = -gt_bboxes_3d[:, -1] - np.pi/2
           
        ### get gt_names
        gt_names = np.array(res['sample_info']['bbox_category'], dtype='<30U')
        res["lidar"]["annotations"] = {
                    "boxes": gt_bboxes_3d.astype(np.float32),
                    "names": gt_names,
                }
        
        return res, info
    
    