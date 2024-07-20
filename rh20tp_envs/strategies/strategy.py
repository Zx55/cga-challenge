from scipy.spatial.transform import Rotation
import torch


def calc_2d_dist(a, b):
    return ((a - b) ** 2).sum().sqrt()


def quat_to_euler(quat, is_degree=False):
    device = quat.device
    r = Rotation.from_quat(quat.cpu().numpy())
    euler = r.as_euler('xyz', degrees=is_degree)
    return torch.from_numpy(euler).to(device)


def calc_rotation_dist(a, b):
    a = a + torch.pi
    if a > torch.pi:
        a = a - 2 * torch.pi
    mi_dist = b - a
    for i in range(3):
        b = b + torch.pi / 2
        if b > torch.pi:
            b = b - 2 * torch.pi
        if b > torch.pi / 2 or b < -torch.pi / 2:
            continue
        if abs(mi_dist) > abs(b - a):
            mi_dist = b - a
    return mi_dist


def wait(action_time_step):
    if action_time_step < 20:
        return torch.tensor([0,0,0,0,0,0,0],dtype=torch.float32)
    return torch.tensor([0,0,0,0,0,0,1],dtype=torch.float32)

    
def pick(
    agent_tcp: torch.Tensor, 
    object_pose: torch.Tensor, 
    last_grasp_time=0, 
    xy_thres=0.005,
    rot_thres=0.1,
    z_thres=0.01,
):
    xy_2d_dist = calc_2d_dist(agent_tcp[:2], object_pose[:2])
    if xy_2d_dist > xy_thres and last_grasp_time < 0:
        target_pose = object_pose[:2] - agent_tcp[:2]
        target_pose = (target_pose / xy_2d_dist) * min(xy_2d_dist*6, 0.1)
        return torch.tensor([target_pose[0],target_pose[1],0,0,0,0,1],dtype=torch.float32)
    
    agent_quat = quat_to_euler(agent_tcp[3:])[0]
    object_quat = quat_to_euler(object_pose[3:])[0]
    if abs(calc_rotation_dist(agent_quat, object_quat)) > rot_thres and last_grasp_time < 0:
        target_pose = calc_rotation_dist(agent_quat, object_quat)
        target_pose = (target_pose / abs(target_pose)) * min(abs(target_pose)*6, 0.2)
        return torch.tensor([0,0,0,0,0,target_pose,1],dtype=torch.float32)
    
    if agent_tcp[2] - object_pose[2] > z_thres and last_grasp_time < 0:
        target_pose = object_pose[2] - agent_tcp[2]
        target_pose = (target_pose / abs(target_pose)) * min(abs(target_pose)*6, 0.1)
        return torch.tensor([0,0,target_pose,0,0,0,1],dtype=torch.float32)
    
    return torch.tensor([0,0,0,0,0,0,-1],dtype=torch.float32)


def up(is_grasped=True):
    if bool(is_grasped):
        return torch.tensor([0,0,0.1,0,0,0,-1],dtype=torch.float32)
    return torch.tensor([0,0,0.1,0,0,0,1],dtype=torch.float32)


def down(is_grasped=True):
    if bool(is_grasped):
        return torch.tensor([0,0,-0.1,0,0,0,-1],dtype=torch.float32)
    return torch.tensor([0,0,-0.1,0,0,0,1],dtype=torch.float32)


def translation(x=0, y=0, is_grasped=True):
    if bool(is_grasped):
        return torch.tensor([x,y,0,0,0,0,-1],dtype=torch.float32)
    return torch.tensor([x,y,0,0,0,0,1],dtype=torch.float32)


def place(
    agent_tcp: torch.Tensor, 
    object_pose: torch.Tensor, 
    xy_thres=0.005, 
    rot_thres=0.02, 
    z_thres=0.075,
):
    xy_2d_dist = calc_2d_dist(agent_tcp[:2], object_pose[:2])
    if xy_2d_dist > xy_thres:
        target_pose = object_pose[:2] - agent_tcp[:2]
        target_pose = (target_pose / xy_2d_dist) * min(xy_2d_dist*6, 0.1)
        return torch.tensor([target_pose[0],target_pose[1],0,0,0,0,-1],dtype=torch.float32)
    
    agent_quat = quat_to_euler(agent_tcp[3:])[0]
    object_quat = quat_to_euler(object_pose[3:])[0]
    if abs(calc_rotation_dist(agent_quat, object_quat)) > rot_thres:
        target_pose = calc_rotation_dist(agent_quat, object_quat)
        target_pose = (target_pose / abs(target_pose)) * min(abs(target_pose)*6, 0.2)
        return torch.tensor([0,0,0,0,0,target_pose,-1],dtype=torch.float32)
    
    if agent_tcp[2] - object_pose[2] > z_thres:
        target_pose = object_pose[2] - agent_tcp[2]
        target_pose = (target_pose / abs(target_pose)) * min(abs(target_pose)*6, 0.1)
        return torch.tensor([0,0,target_pose,0,0,0,-1],dtype=torch.float32)
    
    return torch.tensor([0,0,0,0,0,0,1],dtype=torch.float32)


def moveontopof(
    agent_tcp: torch.Tensor, 
    object_pose: torch.Tensor, 
    xy_thres=0.005, 
    rot_thres=0.02, 
    z_thres=0.075,
):
    xy_2d_dist = calc_2d_dist(agent_tcp[:2], object_pose[:2])
    if xy_2d_dist > xy_thres:
        target_pose = object_pose[:2] - agent_tcp[:2]
        target_pose = (target_pose / xy_2d_dist) * min(xy_2d_dist*6, 0.1)
        return torch.tensor([target_pose[0],target_pose[1],0,0,0,0,-1],dtype=torch.float32)
    
    agent_quat = quat_to_euler(agent_tcp[3:])[0]
    object_quat = quat_to_euler(object_pose[3:])[0]
    if abs(calc_rotation_dist(agent_quat, object_quat)) > rot_thres:
        target_pose = calc_rotation_dist(agent_quat, object_quat)
        target_pose = (target_pose / abs(target_pose)) * min(abs(target_pose)*6, 0.2)
        return torch.tensor([0,0,0,0,0,target_pose,-1],dtype=torch.float32)
    
    if agent_tcp[2] - object_pose[2] > z_thres:
        target_pose = object_pose[2] - agent_tcp[2]
        target_pose = (target_pose / abs(target_pose)) * min(abs(target_pose)*6, 0.1)
        return torch.tensor([0,0,target_pose,0,0,0,-1],dtype=torch.float32)
    
    return None


InitPose = torch.tensor([ 0.0120,  0.0141,  0.1840, -0.0012,  0.9999, -0.0032,  0.0143])
def init(agent_tcp, gripper=0, cfg=[0.05]):
    init_pos = quat_to_euler(InitPose[3:])
    agent_pos = quat_to_euler(agent_tcp[3:])
    if abs(init_pos[1] - agent_pos[1]) > cfg[0]:
        target_pose = init_pos[1] - agent_pos[1]
        target_pose = (target_pose / abs(target_pose)) * min(abs(target_pose)*6, 0.2)
        return torch.tensor([0,0,0,0,target_pose,0,gripper],dtype=torch.float32)
    if abs(init_pos[2] - agent_pos[2]) > cfg[0]:
        target_pose = init_pos[2] - agent_pos[2]
        target_pose = (target_pose / abs(target_pose)) * min(abs(target_pose)*6, 0.2)
        return torch.tensor([0,0,0,target_pose,0,0,gripper],dtype=torch.float32)
    return None
