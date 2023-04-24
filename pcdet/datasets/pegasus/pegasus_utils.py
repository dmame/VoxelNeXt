import os
import pickle

import numpy as np

def get_obj(path):
    with open(path, 'rb') as f:
            obj = pickle.load(f)
    return obj 

def read_single_pegasus_bin(bin_path):
    """get point from path, abs path, bin file"""
    points = np.fromfile(bin_path, dtype=np.float32)
    
    # x,y,z,i,t
    # i, normalized
    # t, microsecond
    points = points.reshape(-1, 4)
    times = np.ones((points.shape[0], 1))
    return points


def read_single_pegasus(obj):
    points_xyz = obj[0]["lidars"]["points_xyz"]
    points_feature = obj[0]["lidars"]["points_feature"]

    # normalize intensity 
    # points_feature[:, 0] = np.tanh(points_feature[:, 0])
    points_feature[:, 0] = points_feature[:, 0] / 255.0

    points = np.concatenate([points_xyz, points_feature[:, 0:1]], axis=-1)
    
    return points 

def read_single_pegasus_sweep(sweep, sweep_idx, time_lag_frep=0.1, nsweeps=2, root_path=None):
    if root_path is not None:
        sweep_path = os.path.join(root_path, sweep['rel_lidar_path'])
    else:
        sweep_path = sweep['path']  ###lidar info pkl path
    
    obj = get_obj(sweep_path)

    points_xyz = obj[0]["lidars"]["points_xyz"]
    points_feature = obj[0]["lidars"]["points_feature"]

    # normalize intensity 
    # points_feature[:, 0] = np.tanh(points_feature[:, 0])
    points_feature[:, 0] = points_feature[:, 0] / 255.0
    points_sweep = np.concatenate([points_xyz, points_feature[:, 0:1]], axis=-1).T # 5 x N

    nbr_points = points_sweep.shape[1]

    if sweep["transform_matrix"] is not None:
        points_sweep[:3, :] = sweep["transform_matrix"].dot( 
            np.vstack((points_sweep[:3, :], np.ones(nbr_points)))
        )[:3, :]
        
    # curr_times = sweep["time_lag"] * np.ones((1, points_sweep.shape[1]))
    ### make the time offset is an constant value
    assert sweep['interval_id'] <= nsweeps - 1, 'the interval_id of sweep must be <= nsweeps - 1' 
    curr_times = sweep['interval_id'] * time_lag_frep * np.ones((1, points_sweep.shape[1]))
    
    return points_sweep.T, curr_times.T


class LoadPointCloudFromFile(object):
    def __init__(self, dataset="PegasusDataset", time_lag_frep=0.1, root_path=None, **kwargs):
        self.type = dataset
        self.random_select = kwargs.get("random_select", False)
        self.npoints = kwargs.get("npoints", 16834)
        self.time_lag_frep = time_lag_frep
        self.root_path = root_path

    def __call__(self, res, info):

        res["type"] = self.type
        
        if self.type == "PegasusDataset":
            if self.root_path is not None:
                path = os.path.join(self.root_path, info['rel_lidar_path'])
            else:
                # lidar path for lidar, or path for lidar
                if 'lidar_path' in info:
                    path = info['lidar_path']
                else:
                    path = info['path']  ###lidar info pkl path
            nsweeps = res["lidar"]["nsweeps"]
            if path.endswith('bin'):
                points = read_single_pegasus_bin(path)
            elif path.endswith('pkl'):
                obj = get_obj(path)
                points = read_single_pegasus(obj)
            else:
                raise NotImplementedError
            
            res["lidar"]["points"] = points
            
            if res['metadata']['num_point_features'] == 5 and nsweeps == 1:
                times = np.zeros_like(points[:, 0:1])
                res["lidar"]["combined"] = np.hstack([points, times]) 

            if nsweeps > 1: 
                sweep_points_list = [points]
                sweep_times_list = [np.zeros((points.shape[0], 1))]

                assert (nsweeps - 1) == len(
                    info["sweeps"]
                ), "nsweeps {} should be equal to the list length {}.".format(
                    nsweeps, len(info["sweeps"])
                )

                for i in range(nsweeps - 1):
                    sweep = info["sweeps"][i]
                    points_sweep, times_sweep = read_single_pegasus_sweep(sweep, i, time_lag_frep=self.time_lag_frep, nsweeps=nsweeps, root_path=self.root_path)
                    sweep_points_list.append(points_sweep)
                    sweep_times_list.append(times_sweep)

                points = np.concatenate(sweep_points_list, axis=0)
                times = np.concatenate(sweep_times_list, axis=0).astype(points.dtype)

                res["lidar"]["points"] = points
                res["lidar"]["times"] = times
                res["lidar"]["combined"] = np.hstack([points, times])
        else:
            raise NotImplementedError

        return res, info

class LoadPointCloudAnnotations(object):
    def __init__(self, with_bbox=True, **kwargs):
        pass

    def __call__(self, res, info):

        if res["type"] == 'PegasusDataset' and "gt_boxes" in info:
            if 'seg_label' in info:
                    seg_label = read_seg_label(info['seg_label'])
                    res["seg_label"] = seg_label
                    res["seg_dataset"] = info['seg_dataset']
            if info["gt_boxes"] is not None:
                res["lidar"]["annotations"] = {
                    "boxes": info["gt_boxes"].astype(np.float32),
                    "names": info["gt_names"],
                }
            
        else:
            pass 

        return res, info