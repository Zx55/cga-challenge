from collections import defaultdict
import io
import math
import numpy as np
import os
import pickle
from PIL import Image
import torch
import torch.distributed as dist
from torch.utils.data import Dataset
from tqdm import tqdm

try:
    from .rh20t_api import (
        binary_search_closest,
        get_tcp_aligned,
        get_gripper,
        check_same_target_sequence,
        homo_2d_to_2d,
        get_camera_params
    )
except:
    # for debug
    import sys
    sys.path.append(os.path.dirname(__file__))
    from rh20t_api import (
        binary_search_closest,
        get_tcp_aligned,
        get_gripper,
        check_same_target_sequence,
        homo_2d_to_2d,
        get_camera_params
    )

try: 
    from petrel_client.client import Client
except: 
    Client = None


def rank0_print(*args, **kwargs):
    if dist.is_initialized() and dist.get_rank() == 0:
        print(*args, **kwargs)
    else:
        print(*args, **kwargs)


def pos2lang(pos):
    x = round(pos[0], 3)
    y = round(pos[1], 3)
    d = round(pos[2], 3)
    return f'({x}, {y}, {d})'


def get_ee_pos(metadata, timestamp, gripper_type='command'):
    # we use quaternion directly here, if you want to transfer quat to rot matrix,
    # you can try (extrinsic @ pose_array_quat_2_matrix(tcp))[:3, :]
    ee_tcp = np.array(get_tcp_aligned(metadata, timestamp))
    ee_gripper = np.array(get_gripper(metadata, timestamp, gripper_type)) / 255.
    ee_pos = np.concatenate([ee_tcp, [ee_gripper]], axis=0)
    return ee_pos


def get_ee_pos_2d(metadata, timestamp, serial):
    intrisics, tcp2cam = get_camera_params(metadata, timestamp, serial, False)
    pt_2d = homo_2d_to_2d(intrisics @ tcp2cam @ np.array([0., 0., 0., 1.], dtype=np.float64).reshape(4, 1))
    x, y = pt_2d[0] / 640, pt_2d[1] / 360
    d = tcp2cam[2][3]
    return np.array([x, y, d])


EE_POS_MIN = np.asarray([-0.57257075, -0.61797443, -0.33872867])
EE_POS_MAX = np.asarray([1.24384486, 0.41246224, 1.72241654])


class LazyRH20TDataset(Dataset):
    valid_keys = ['image', 'depth', 'image_timestamp', 'action_labels']

    def __init__(
        self, 
        data_root: str, 
        anno_path: str,
        use_multi_view=False,
        use_multi_frame=False,
        multi_frame_n_frame=0,
        multi_frame_n_time=0,
        use_ceph=False,
        use_cache=False,
    ):
        '''
        Basic API for RH20T, containing multi-view/multi-frame image/depth data

        Args:
            data_root: /path/to/RH20T
            anno_path: /path/to/RH20TP/annotation, pickle file
            use_multi_view: if True, will return a multi-view image/depth dict, 
                            see `self.get_multi_view_image_dict(...)`
            use_multi_frame: if True, will return a multi-frame image/depth dict,
                             see `self.get_multi_frame_data_dict(...)`
            multi_frame_n_frame: how many extra frames are used in multi-frame setup
            multi_frame_n_time: total duration (second) of sampled multi-frame, 
                                interval of multi-frame = multi_frame_n_time / multi_frame_n_frame
            use_ceph: if use ceph client
            use_cache: save processed data to ~/.cache/rh20tp/ if cache file not exist,
                       read cache file for fast loading if exist.
        '''
        super().__init__()
        self.data_path = os.path.join(data_root, 'RH20T_cfg{}/{}/cam_{}/{}/{}.{}')
        self.use_multi_view = use_multi_view
        self.use_multi_frame = use_multi_frame
        self.multi_frame_n_frame = multi_frame_n_frame
        self.multi_frame_n_time = multi_frame_n_time

        self.data_types = ['color', 'depth']
        self.use_cache = use_cache

        self.client = self.init_client(use_ceph)
        self.anno = self.read_anno(anno_path)

    def init_client(self, use_ceph):
        if not use_ceph:
            return None
    
        if Client is None:
            raise RuntimeError(f'Ceph client is inavailable')
        
        return Client('~/petreloss.conf')
    
    def get_multi_view_image_dict(
        self, 
        task_id: str, 
        pick_timestamp,
        cameras,
    ):
        """
        construct multi-view image dict

        Args:
            cameras: a dictionary that stores color/depth/tcp/gripper timestamps of different camera,
                     i.e., its format is like:
                     {
                         'cam_xxx': {
                            'img_timestamps': [xxx, xxx],  # timestamp
                            ...
                         },
                         ...
                     }
        
        Return: multi-view color/depth image dictionary of searched_timestamp that is closest to pick_timestamp,
                i.e., its format is like:
                {
                    'color': {
                        'cam_xxx': [image_searched_timestamp_camera_i],
                        ...
                    },
                    'depth': {
                        'cam_xxx': [depth_searched_timestamp_camera_i],
                        ...
                    }
                }  
        """
        cfg_id = int(task_id.split('_')[-1].strip())

        image_dict = defaultdict(dict)
        for camera_id, camera_metadata in cameras.items():
            searched_timestamp = binary_search_closest(
                camera_metadata['img_timestamps'], pick_timestamp)
            searched_timestamp = int(searched_timestamp)
            for image_type in self.data_types:
                image_path = self.data_path.format(
                    cfg_id, task_id, camera_id, image_type, f'{searched_timestamp:d}',
                    'jpg' if image_type == 'color' else 'png')
                image_dict[image_type][camera_id] = [image_path]
                image_dict[f'{image_type}_timestamp'][camera_id] = [searched_timestamp]
        return image_dict
    
    def get_multi_frame_data_dict(
        self, 
        data_dicts,
        metadata,
    ):
        """
        Args: multi-view data dict of all sampled frames,
            i.e., its format is like:
            [
                {
                    'image': {
                        'cam_xxx': [image_searched_timestamp_camera_i],
                        ...
                    },
                    ...
                }
            ]
        Return: multi-view and multi-frame data dict,
            i.e., its format is like:
            [
                {
                    'image': {
                        'cam_xxx': [image_tn-k, image_tn-k+1, ..., image_tn],
                        ...
                    },
                    ...
                }
            ]
        """
        multi_frame_len = self.multi_frame_n_frame
        multi_frame_time = self.multi_frame_n_time
        if multi_frame_len == 0 or multi_frame_time == 0:
            return data_dicts
        timestamp_gap = multi_frame_time*1000//multi_frame_len
        
        start_timestamp = metadata['base_aligned_timestamps'][0]
        for i, data_dict in enumerate(data_dicts):
            end_timestamp = math.ceil(data_dict['aligned_timestamp'])
            history_timestamps = np.linspace(
                start_timestamp, 
                end_timestamp, 
                (end_timestamp - start_timestamp) // timestamp_gap).tolist()
            history_timestamps = [start_timestamp] * (multi_frame_len - len(history_timestamps))\
                    + history_timestamps # using first image to extend the history

            for camera in data_dict['image'].keys():
                history_image_list = []
                history_image_timestamp_list = []
                history_depth_list = []
                
                for timestamp in history_timestamps[-multi_frame_len:-1]:
                    searched_timestamp = binary_search_closest(
                        metadata['camera'][camera]['img_timestamps'], timestamp, should_lower=True)
                    searched_timestamp = int(searched_timestamp)
                    raw_image_path = data_dict['image'][camera][-1]
                    new_image_file = f'{searched_timestamp:d}.jpg'
                    image_path = raw_image_path[:-len(new_image_file)] + new_image_file
                    history_image_list.append(image_path)
                    history_image_timestamp_list.append(searched_timestamp)
                        
                    raw_depth_path = data_dict['depth'][camera][-1]
                    new_depth_file = f'{searched_timestamp:d}.png'
                    depth_path = raw_depth_path[:-len(new_depth_file)] + new_depth_file
                    history_depth_list.append(depth_path)
                
                image_list = history_image_list + data_dicts[i]['image'][camera]
                depth_list = history_depth_list + data_dicts[i]['depth'][camera]
                image_timestamp_list = history_image_timestamp_list + data_dicts[i]['image_timestamp'][camera]
                data_dicts[i]['image'][camera] = image_list
                data_dicts[i]['depth'][camera] = depth_list
                data_dicts[i]['image_timestamp'][camera] = image_timestamp_list
                
                # check
                if len(image_timestamp_list) > 1:
                    image_timestamp_list = torch.as_tensor(image_timestamp_list)
                    diff = (image_timestamp_list[1:] - image_timestamp_list[:-1])
                    assert (diff >= 0).all().item() is True, f'{image_timestamp_list}'
        return data_dicts
    
    def get_cameras(self, metadata):
        primary_camera = sorted(list(metadata['camera'].keys()))[0]
        if self.use_multi_view:
            cameras = metadata['camera']
        else:
            cameras = {primary_camera: metadata['camera'][primary_camera]}
        return cameras, primary_camera

    def get_cached_file(self):
        raise NotImplementedError
    
    def save_cached_file(self, data):
        if not self.use_cache:
            return
        
        if (not dist.is_initialized()) or dist.get_rank() == 0:
            cached_dir, cached_file = self.get_cached_file()
            os.makedirs(cached_dir, exist_ok=True)

            with open(cached_file, 'wb') as f:
                pickle.dump(data, f)
            rank0_print(f'save cached file to {cached_file}')
        if dist.is_initialized():
            dist.barrier()

    def read_cached_file(self):
        if not self.use_cache:
            return None

        _, cached_file = self.get_cached_file()
        if os.path.exists(cached_file):
            rank0_print(f'read cached file from {cached_file}')
            with open(cached_file, 'rb') as f:
                cached_file = pickle.load(f)
            return cached_file
        return None

    def read_metadata(self, metadata, task_id, cameras, primary_camera):
        raise NotImplementedError

    def read_anno(self, anno_path):
        cached_file = self.read_cached_file()
        if cached_file is not None:
            return cached_file

        if anno_path.endswith('.pkl'):
            with open(anno_path, 'rb') as f:
                metadata = pickle.load(f)
        else:
            raise ValueError(f'only support pickle file')
        
        annos = []
        for metadata_i in tqdm(metadata, desc=f'build {self.__class__.__name__}'):
            task_id = metadata_i['demo_folder']
            cameras, primary_camera = self.get_cameras(metadata_i)
            data_dicts = self.read_metadata(metadata_i, task_id, cameras, primary_camera)
            annos.extend(data_dicts)

        self.save_cached_file(annos)
        rank0_print(f'totally {len(metadata)} demonstrations, and {len(annos)} samples')
        return annos
    
    def read_image(self, image_file: str):
        if image_file.startswith('s3://'):
            assert self.client is not None, f'client is not initilized'
            image_bytes = self.client.get(image_file)
            assert image_bytes is not None, f'data path {image_file} is invalid'
            byte_stream = io.BytesIO(image_bytes)
            image = Image.open(byte_stream).convert('RGB')
        
        else:
            image = Image.open(image_file).convert('RGB')
        return image
    
    def __getitem__(self, index):
        '''
        Return:
            image: PIL Image, multi-view and multi-frame RGB images
            depth: PIL Image, multi-view and multi-frame depth images
            image_timestamp: timestamps of multi-view/multi-frame images, can be used as 
                             temporal embedding for each image
            action_labels: see `self.get_action_labels(...)`
        '''
        anno = self.anno[index]

        image_file = anno['image']
        if isinstance(image_file, str):
            anno['image'] = self.read_image(image_file)
        else:
            anno['image'] = {cam: [self.read_image(f) for f in historical_frames]
                             for cam, historical_frames in image_file.items()}
            
        depth_file = anno['depth']
        if isinstance(image_file, str):
            anno['depth'] = self.read_image(depth_file)
        else:
            anno['depth'] = {cam: [self.read_image(f) for f in historical_frames]
                             for cam, historical_frames in depth_file.items()}
            
        new_anno = {}
        for key in self.valid_keys:
            new_anno[key] = anno[key]
        return new_anno
    

class LazyRH20TPrimitiveDataset(LazyRH20TDataset):

    def __init__(
        self, 
        data_root: str, 
        anno_path: str, 
        sample_rate: int = 5,
        use_multi_view=False, 
        use_multi_frame=False, 
        multi_frame_n_frame=0, 
        multi_frame_n_time=0, 
        use_ceph=False,
        use_cache=False,
    ):
        '''
        API for RH20T-P dataset, containing primitive-level information

        Args:
            sample_rate: how many frames are sampled per second when constructing the trajectory
        '''
        self.sample_rate = sample_rate
        super().__init__(data_root, anno_path, use_multi_view, use_multi_frame, 
                         multi_frame_n_frame, multi_frame_n_time, use_ceph, use_cache)

    def get_cached_file(self):
        cached_dir = os.path.join(os.path.expanduser('~'), '.cache', 'rh20tp')
        cached_file = os.path.join(cached_dir, f'rh20tp_primitive_cache.pkl')
        return cached_dir, cached_file

    def get_action_labels(self, metadata, action_list, serial, index, normalized=False):
        '''
        Return:
            current_action: str, current primitive skill
            historical_actions: List[str], completed primitive skills in this task
            cur_ee_pos: np.ndarray, current end-effector (gripper) 3D coordinate (xyz + quat rot + gripper 
                         width) under world coordinate system
            cur_ee_pos_2d: np.ndarray, current end-effector (gripper) 2D coordinate (x, y, d) under image 
                           coordinate system
            target_ee_pos: np.ndarray, GT end-effector (gripper) 3D coordinate (xyz + quat rot + gripper 
                           width) under world coordinate system
            target_ee_pos_2d: np.ndarray, GT end-effector (gripper) 2D coordinate (x, y, d) under image 
                           coordinate system
            trajectory_ee: np.ndarray, 3D trajectory formed by the end-effector (gripper) 3D coordinate (xyz
                           + quat rot + gripper width) from start_timestamp to end_timestamp
            trajectory_ee-2d: np.ndarray, 2D trajectory formed by the end-effector (gripper) 2D coordinate (
                              x, y, d) from start_timestamp to end_timestamp
        '''
        # actions (historical + current)
        historical_actions = [
            action_list[j]['action_description'].lower().replace(
                '{pos}', pos2lang(action_list[j]['end_arm_pos']))
            for j in range(index)
        ]

        start_timestamp = action_list[index]['start_timestamp']
        end_timestamp = action_list[index]['end_timestamp']

        current_action = action_list[index]['action_description'].lower()
        if current_action == 'done':
            action_list[index]['start_arm_pos'] = action_list[index - 1]['end_arm_pos']
        
        cur_ee_pos = get_ee_pos(metadata, start_timestamp, 'info')
        cur_ee_pos_2d = get_ee_pos_2d(metadata, start_timestamp, serial)
        target_ee_pos = get_ee_pos(metadata, end_timestamp, 'command')
        target_ee_pos_2d = get_ee_pos_2d(metadata, end_timestamp, serial)

        if normalized:
            cur_ee_pos[:3] = (cur_ee_pos[:3] - EE_POS_MIN) / (EE_POS_MAX - EE_POS_MIN)
            target_ee_pos[:3] = (target_ee_pos[:3] - EE_POS_MIN) / (EE_POS_MAX - EE_POS_MIN)

        # generate trajectory
        if current_action == 'done':
            trajectory_ee, trajectory_ee_2d = None, None
        else:
            num_samples = max((end_timestamp - start_timestamp) // (1000 // self.sample_rate), 1)
            trajectory_timestamps = np.linspace(start_timestamp, end_timestamp, num_samples)

            trajectory_ee, trajectory_ee_2d = [], []
            for timestamp in trajectory_timestamps:
                ee_pos = get_ee_pos(metadata, timestamp, 'command')
                if normalized:
                    ee_pos[:3] = (ee_pos[:3] - EE_POS_MIN) / (EE_POS_MAX - EE_POS_MIN)
                trajectory_ee.append(ee_pos)

                ee_pos_2d = get_ee_pos_2d(metadata, timestamp, serial)
                trajectory_ee_2d.append(ee_pos_2d)

            trajectory_ee = np.vstack(trajectory_ee)
            trajectory_ee_2d = np.vstack(trajectory_ee_2d)
            
        return dict(
            current_action=current_action,
            historical_actions=historical_actions,
            cur_ee_pos_2d=cur_ee_pos_2d,
            cur_ee_pos=cur_ee_pos,
            target_ee_pos_2d=target_ee_pos_2d,
            target_ee_pos=target_ee_pos,
            trajectory_ee=trajectory_ee,
            trajectory_ee_2d=trajectory_ee_2d,
        )
    
    def read_metadata(self, metadata, task_id, cameras, primary_camera):
        action_list = metadata['action_list']
        pick_timestamps = [action['start_timestamp'] for action in action_list]

        data_dicts = []
        for i, timestamp in enumerate(pick_timestamps):
            action_labels = self.get_action_labels(
                metadata, action_list, primary_camera, index=i)
            image_dict = self.get_multi_view_image_dict(
                task_id, timestamp, cameras)
            data_dict = dict(
                image=image_dict['color'],
                depth=image_dict['depth'],
                image_timestamp=image_dict['color_timestamp'],
                action_labels=action_labels,
                aligned_timestamp=timestamp
            )
            data_dicts.append(data_dict)

        if self.use_multi_frame:
            data_dicts = self.get_multi_frame_data_dict(data_dicts, metadata)
        
        return data_dicts


class LazyRH20TActionDataset(LazyRH20TDataset):

    def __init__(self, 
        data_root: str, 
        anno_path: str, 
        sample_rate: int = 5,
        sample_n_time: int = 1,
        use_multi_view=False, 
        use_multi_frame=False, 
        multi_frame_n_frame=0, 
        multi_frame_n_time=0,
        use_ceph=False,
        use_cache=False,
    ):
        '''
        API for RH20T dataset, containing actions at each timestamp

        Args:
            sample_n_time: action labels will contain the actions sampled from current 
                           timestamp to `sample_n_time` (second)
            sample_rate: how many frames are sampled per second between current timestamp
                         to `sample_n_time`
        '''
        self.sample_rate = sample_rate
        self.sample_n_time = sample_n_time
        super().__init__(data_root, anno_path, use_multi_view, use_multi_frame, 
                         multi_frame_n_frame, multi_frame_n_time, use_ceph, use_cache)
        
    def get_cached_file(self):
        cached_dir = os.path.join(os.path.expanduser('~'), '.cache', 'rh20tp')
        cached_file = os.path.join(cached_dir, f'rh20tp_action_cache.pkl')
        return cached_dir, cached_file

    def get_action_labels(self, metadata, cur_timestamp, serial, normalized=False):
        '''
        Return:
            cur_ee_pos: np.ndarray, current end-effector 3D coordinate (xyz + quat rot + gripper width) under world 
                        coordinate system
            cur_ee_pos_2d: np.ndarray, current end-effector 2D coordinate (x, y, d) under image coordinate system
            target_ee_pos: np.ndarray, GT end-effector 3D coordinate (xyz + quat rot + gripper width) from current 
                           timestamp to `sample_n_time` under world coordinate system
            target_ee_pos_2d: np.ndarray, GT end-effector 2D coordinate (x, y, d) from current timestamp to 
                              `sample_n_time` under image coordinate system
        '''
        cur_ee_pos = get_ee_pos(metadata, cur_timestamp, 'info')
        cur_ee_pos_2d = get_ee_pos_2d(metadata, cur_timestamp, serial)

        # sample target frames
        end_timestamp = int(cur_timestamp + self.sample_n_time * (1000 // self.sample_rate))
        target_regular_timestamps = np.linspace(
            cur_timestamp, 
            end_timestamp, 
            self.sample_n_time + 1).tolist()[1:]
        assert len(target_regular_timestamps) == self.sample_n_time

        target_ee_pos, target_ee_pos_2d = [], []
        for timestamp in target_regular_timestamps:
            target_ee_pos.append(get_ee_pos(metadata, timestamp, 'command'))
            target_ee_pos_2d.append(get_ee_pos_2d(metadata, timestamp, serial))
        target_ee_pos = np.vstack(target_ee_pos)
        target_ee_pos_2d = np.vstack(target_ee_pos_2d)

        # filter standby actions
        if check_same_target_sequence(cur_ee_pos, target_ee_pos):  
            return None        

        # normalization
        if normalized:
            cur_ee_pos[:3] = (cur_ee_pos[:3] - EE_POS_MIN) / (EE_POS_MAX - EE_POS_MIN)
            target_ee_pos[:, :3] = (target_ee_pos[:, :3] - EE_POS_MIN) / (EE_POS_MAX - EE_POS_MIN)

        return dict(
            cur_ee_pos=cur_ee_pos,
            cur_ee_pos_2d=cur_ee_pos_2d,
            target_ee_pos=target_ee_pos[:self.sample_n_time, :],
            target_ee_pos_2d=target_ee_pos_2d[:self.sample_n_time, :]
        )

    def read_metadata(self, metadata, task_id, cameras, primary_camera):
        base_aligned_timestamps = metadata['base_aligned_timestamps']
        start_timestamp = base_aligned_timestamps[0]
        end_timestamp = base_aligned_timestamps[-1]
        num_samples = max((end_timestamp - start_timestamp) // (1000 // self.sample_rate), 1)
        pick_timestamps = np.linspace(start_timestamp, end_timestamp, num_samples)
        
        data_dicts = []
        for timestamp in pick_timestamps:
            action_labels = self.get_action_labels(metadata, timestamp, primary_camera)
            if action_labels is None:
                continue

            image_dict = self.get_multi_view_image_dict(
                task_id, timestamp, cameras)
            data_dict = dict(
                image=image_dict['color'],
                depth=image_dict['depth'],
                image_timestamp=image_dict['color_timestamp'],
                action_labels=action_labels,
                aligned_timestamp=timestamp
            )
            data_dicts.append(data_dict)

        if self.use_multi_frame:
            data_dicts = self.get_multi_frame_data_dict(data_dicts, metadata)
        
        return data_dicts


if __name__ == '__main__':
    data = LazyRH20TPrimitiveDataset(
        data_root='',
        anno_path='../sources/rh20tp_cga_metadata_v1.0.pkl',
        use_multi_view=True,
        use_multi_frame=True,
        multi_frame_n_frame=3,
        multi_frame_n_time=1,
        use_cache=True
    )

    d = data[0]