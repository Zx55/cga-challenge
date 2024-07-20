from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils.building.ground import build_ground
from mani_skill.utils.scene_builder import SceneBuilder
import numpy as np
import os
import sapien
import sapien.render
import torch
from transforms3d.euler import euler2quat
from typing import List

from rh20tp_envs.loaders.tables.utils import init_table_robot


class Rh20tTableSceneBuilder(SceneBuilder):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.assets_dir = None

    def build(self):
        model_file = os.path.join(self.assets_dir, "textured.dae")
        collision_file = os.path.join(self.assets_dir, "collision.obj") 
        pose = sapien.Pose(q=euler2quat(0, 0, np.pi / 2))
        scale = [1.5, 2.0, 1.75]
        
        builder = self.scene.create_actor_builder()
        builder.add_nonconvex_collision_from_file(
            filename=collision_file,
            scale=scale,
            pose=pose)
        builder.add_visual_from_file(
            filename=model_file, 
            scale=scale, 
            pose=pose)
        table = builder.build_kinematic(name="table-workspace")

        aabb = table._objs[0].find_component_by_type(
            sapien.render.RenderBodyComponent).compute_global_aabb_tight()
    
        self.table_length = aabb[1, 0] - aabb[0, 0]
        self.table_width = aabb[1, 1] - aabb[0, 1]
        self.table_height = aabb[1, 2] - aabb[0, 2]
        self.ground = build_ground(self.scene, altitude=-self.table_height)
        self.table = table
        self.scene_objects: List[sapien.Entity] = [self.table, self.ground]

    def initialize(self, env_idx: torch.Tensor):
        b = len(env_idx)
        pose = sapien.Pose(p=[-0.25, 0, -self.table_height + 0.7], q=euler2quat(0, 0, np.pi / 2))
        self.table.set_pose(pose)
        init_table_robot(b, self.env, self.ground, self.robot_init_qpos_noise, self.table_height)
