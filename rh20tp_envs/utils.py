from mani_skill.utils.geometry.trimesh_utils import get_component_mesh
from mani_skill.utils.structs import Actor, Articulation
import numpy as np
import torch
from trimesh.primitives import Box
import random
import sapien.physx as physx
from transforms3d.euler import euler2quat
from typing import List, Union


def random_signs(b: int, return_torch=False):
    sign = np.random.choice([1, -1], (b,), replace=True)
    if return_torch:
        return torch.from_numpy(sign)
    else:
        return sign


def random_quats_along_z(angle_choices: List[float], b: int, noise_scale=0.2):
    rot_angle = np.random.choice(angle_choices, (b,), replace=True)
    noise_sign = random_signs(b)
    noise = np.random.rand(b) * noise_sign * noise_scale
    rot_angle = rot_angle + noise

    rot_quat = [torch.from_numpy(euler2quat(0, 0, angle)) for angle in rot_angle]
    return torch.stack(rot_quat).to(torch.float32)


def random_quats_along_xy(angle_choices: List[float], b: int, noise_scale=0.2):
    rot_angle = np.random.choice(angle_choices, (b,), replace=True)
    noise_sign = random_signs(b)
    noise = np.random.rand(b) * noise_sign * noise_scale
    rot_angle = rot_angle + noise

    rot_direction = np.random.choice([0, 1], (b,), replace=True)
    rot_euler = [[rot_angle[i], 0, 0] if rot_direction[i] == 0 else [0, rot_angle[i], 0] 
                 for i in range(b)]
    rot_quat = [torch.from_numpy(euler2quat(euler[0], euler[1], euler[2])) 
                for euler in rot_euler]
    return torch.stack(rot_quat).to(torch.float32)


def random_xys_avoid_obstacle(obstacle, b, x_scale, x_trans, y_scale, y_trans, avoid_scale, avoid_trans):
    device = obstacle.pose.p.device
    vertices = obstacle.get_collision_meshes()[0].vertices
    range_xy = torch.from_numpy(np.array(vertices.max(axis=0) - vertices.min(axis=0))[:2]).abs()
    sign = random_signs(b, return_torch=True).to(device)

    xy = torch.zeros((b, 2))
    if range_xy[0] > range_xy[1]:
        xy[:, 0] = torch.rand((b,)) * x_scale + x_trans
        xy[:, 1] = obstacle.pose.p[:, 1] + sign * (range_xy[1] * 0.5 + torch.rand((b,)) * avoid_scale + avoid_trans)
    else:
        xy[:, 0] = obstacle.pose.p[:, 0] + sign * (range_xy[0] * 0.5 + torch.rand((b,)) * avoid_scale + avoid_trans)
        xy[:, 1] = torch.rand((b,)) * y_scale + y_trans
    return xy


# todo: need test
def random_xys_can_touch(b=1, radius_min=0.3,radius_max=0.8):
    x = torch.rand((b,)) * 0.7 + 0.1
    y = torch.rand((b,)) * 1.6 - 0.8
    for i in range(b):
        xy_dist = (x[i]**2 + y[i]**2).sqrt()
        while xy_dist > radius_max or xy_dist < radius_min:
            x[i] = random.random() * 0.7 + 0.1
            y[i] = random.random() * 1.6 - 0.8
            xy_dist = (x[i]**2 + y[i]**2).sqrt()
    return x, y


def get_actor_obb(actor: Actor, to_world_frame=False):
    b = actor.pose.p.shape[0]

    obbs = []
    for i in range(b):
        mesh = get_component_mesh(
            actor._objs[i].find_component_by_type(physx.PhysxRigidDynamicComponent), 
            to_world_frame=to_world_frame)
        obb: Box = mesh.bounding_box_oriented
        obbs.append(obb)
    return obbs


def get_articulation_obb(articulation: Articulation, name=None, to_world_frame=False):
    b = articulation.pose.p.shape[0]

    obbs = []
    for i in range(b):
        obbs_i = []
        links = articulation._objs[i].get_links()
        for link in links:
            entity = link.get_entity()
            if name is not None and entity.get_name() != name:
                continue

            mesh = get_component_mesh(
                entity.find_component_by_type(physx.PhysxArticulationLinkComponent),
                to_world_frame=to_world_frame)
            obb: Box = mesh.bounding_box_oriented
            obbs_i.append(obb)
        obbs.append(obbs_i)
    return obbs


def get_bbox_from_obb(obb: Union[Box, List[Box]]):
    if isinstance(obb, List):  # articulation
        if len(obb) == 1:
            return get_bbox_from_obb(obb[0])

        boxes = [torch.tensor(obb_i.vertices) for obb_i in obb]
        raise NotImplementedError
    else:
        return torch.tensor(obb.vertices)


def is_obb_a_inside_obb_b(
    obb_a: Union[Box, List[Box]], 
    obb_b: Union[Box, List[Box]], 
    pose_a: torch.Tensor, 
    pose_b: torch.Tensor
):
    bbox_a = get_bbox_from_obb(obb_a) + pose_a
    bbox_b = get_bbox_from_obb(obb_b) + pose_b

    a_min = bbox_a.min(dim=0).values
    a_max = bbox_a.max(dim=0).values
    b_min = bbox_b.min(dim=0).values
    b_max = bbox_b.max(dim=0).values

    return torch.all(a_min >= b_min) and torch.all(a_max <= b_max)
