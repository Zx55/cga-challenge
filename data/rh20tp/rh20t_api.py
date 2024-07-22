"""
    Searching functions for interpolations
"""
import numpy as np
from math import *
from transforms3d.quaternions import quat2mat, mat2quat


def binary_search_latest_range(arr, l:int, r:int, x):
    if arr[r] <= x or r == l: 
        return arr[r]
    mid = ((l + r) >> 1) + 1
    return binary_search_latest_range(arr, mid, r, x) if arr[mid] <= x else binary_search_latest_range(arr, l, mid - 1, x)


def binary_search_latest(arr:list, x):
    '''
        search for the nearest item in arr just smaller than x,
        if no one smaller than x is found, return the smallest
        
        Params:
        ----------
        arr:    the array to search on
        x:      the target value
        
        Returns:
        ----------
        x_t:    the closest previous value in arr of x
    '''
    if len(arr) <= 0: 
        raise ValueError("input array should contain at least one element")
    return binary_search_latest_range(arr, 0, len(arr) - 1, x)


def interpolate_linear(target_t:int, t1:int, t2:int, x1:np.ndarray, x2:np.ndarray): 
    return (x1 + (target_t - t1) / (t2 - t1) * (x2 - x1) if t1 != t2 else x1)


def binary_search_closest_two_idx(t:np.array, target_t:int):
    # linearly searches indices of two closest element of target_t in t
    # for continuous values (i.e. non-image data) searching
    prev_t_idx = np.where(t == (binary_search_latest(t, target_t)))[0][0]
    return (prev_t_idx, (prev_t_idx + 1 if prev_t_idx < len(t) - 1 else prev_t_idx - 1))


def binary_search_closest(t:np.array, target_t:int, should_lower = False):
    # for image path searching
    if target_t in t: 
        return target_t
    prev_t_idx = np.where(t == binary_search_latest(t, target_t))[0][0]
    if prev_t_idx == len(t) - 1: 
        res = t[prev_t_idx]
    else:
        res = t[prev_t_idx] if abs(t[prev_t_idx] - target_t) <= abs(t[prev_t_idx + 1] - target_t) else t[prev_t_idx + 1]
    if should_lower and res > target_t:
        res = t[prev_t_idx]

    return res


def sort_by_timestamp(_dict_of_list_of_dict):
    for _k in _dict_of_list_of_dict: 
        _dict_of_list_of_dict[_k] = sorted(_dict_of_list_of_dict[_k].values(), key=lambda item: item["timestamp"])


def get_tcp_aligned(metadata, pick_timestamp):
    base_aligned_timestamps = metadata['base_aligned_timestamps']
    base_aligned_timestamps_time_serial_idx_tuples = metadata['base_aligned_timestamps_time_serial_idx_tuples']
    _idx_1, _idx_2 = binary_search_closest_two_idx(base_aligned_timestamps, pick_timestamp)
    (time_1, serial_1, serial_idx_1) = base_aligned_timestamps_time_serial_idx_tuples[_idx_1]
    (time_2, serial_2, serial_idx_2) = base_aligned_timestamps_time_serial_idx_tuples[_idx_2]
    return interpolate_linear(
        pick_timestamp,
        base_aligned_timestamps[_idx_1],
        base_aligned_timestamps[_idx_2],
        metadata['camera'][serial_1]['tcp_base'][serial_idx_1],
        metadata['camera'][serial_2]['tcp_base'][serial_idx_2]
    )


def get_gripper(metadata, pick_timestamp, gripper_type='command'):
    base_aligned_timestamps = metadata['base_aligned_timestamps']
    base_aligned_timestamps_time_serial_idx_tuples = metadata['base_aligned_timestamps_time_serial_idx_tuples']
    _idx_1, _idx_2 = binary_search_closest_two_idx(base_aligned_timestamps, pick_timestamp)
    (time_1, serial_1, serial_idx_1) = base_aligned_timestamps_time_serial_idx_tuples[_idx_1] # (timestamp, camera_id, id_in_tcp)
    (time_2, serial_2, serial_idx_2) = base_aligned_timestamps_time_serial_idx_tuples[_idx_2]
    return interpolate_linear(
        pick_timestamp,
        base_aligned_timestamps[_idx_1],
        base_aligned_timestamps[_idx_2],
        metadata['camera'][serial_1][f'gripper_{gripper_type}'][serial_idx_1],
        metadata['camera'][serial_2][f'gripper_{gripper_type}'][serial_idx_2],
    )


def check_same_target_sequence(input_tcp, output_tcp:np.array):
    for item in output_tcp:
        if not np.all(input_tcp == item):
            return False
    return True


def check_same_numpy_dict(np_dict_a, np_dict_b):
    keys = np_dict_a.keys()
    for key in keys:
        if not np.all(np_dict_a[key] == np_dict_b[key]):
            return False
    return True


def pose_array_quat_2_matrix(pose):
    '''transform pose array of quaternion to transformation matrix

    Param:
        pose:   7d vector, with t(3d) + q(4d)
    ----------
    Return:
        mat:    4x4 matrix, with R,T,0,1 form
    '''
    mat = quat2mat([pose[3], pose[4], pose[5], pose[6]])

    return np.array([[mat[0][0], mat[0][1], mat[0][2], pose[0]],
                     [mat[1][0], mat[1][1], mat[1][2], pose[1]],
                     [mat[2][0], mat[2][1], mat[2][2], pose[2]],
                     [0, 0, 0, 1]])


def matrix_2_pose_array_quat(mat):
    '''transform transformation matrix to pose array of quaternion

    Param:
        mat:        4x4 matrix, with R,T,0,1 form
    ----------
    Return:
        pose:       7d vector, with t(3d) + q(4d)
    '''
    rotation_mat = np.array([[mat[0][0], mat[0][1], mat[0][2]],
                             [mat[1][0], mat[1][1], mat[1][2]],
                             [mat[2][0], mat[2][1], mat[2][2]]])
    q = mat2quat(rotation_mat)
    return np.array([mat[0][3], mat[1][3], mat[2][3], q[0], q[1], q[2], q[3]])


def pos_quat_gripper_to_pos_rot6d_gripper(items): 
    outputs = []
    for item in items: 
        pos, pos_quat, gripper = item[:3], item[:7], item[7]
        transform = pose_array_quat_2_matrix(pos_quat)
        rotmat = transform[:3, :3]
        rot6d = rotmat[:, :2].reshape(6, )
        output = np.concatenate([pos, rot6d, [gripper]])
        outputs.append(output)  
    outputs = np.stack(outputs).astype(np.float32)
    return outputs 


def homo_2d_to_2d(tcp_img:np.array, width = 640, height = 360):
    tcp_img[0][0] = tcp_img[0][0] / tcp_img[2][0]
    tcp_img[1][0] = tcp_img[1][0] / tcp_img[2][0]
    tcp_img[2][0] = tcp_img[2][0] / tcp_img[2][0]
    tcp_rgb = np.array([[tcp_img[0][0] / 2],[tcp_img[1][0] / 2]])
    tcp_rgb[0][0] = max(min(tcp_rgb[0][0], width), 0)
    tcp_rgb[1][0] = max(min(tcp_rgb[1][0], height), 0)
    return tuple(tcp_rgb.transpose(1,0)[0].astype(int).tolist())


def get_camera_params(metadata, timestamp, serial, have_transfered=False):
    pred_tcp_camera = get_tcp_aligned(metadata, timestamp)
    if not have_transfered:
        serial_extrinsics = metadata['camera'][serial]['extrinsics']
        pred_tcp_camera = serial_extrinsics @ pose_array_quat_2_matrix(pred_tcp_camera)
    
    intrinsics = metadata['camera'][serial]['intrinsics']
    return intrinsics, pred_tcp_camera
