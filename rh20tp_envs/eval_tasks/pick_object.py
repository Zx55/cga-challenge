import mani_skill.envs.utils.randomization as randomization
from mani_skill.agents.robots import Fetch, Panda, Xmate3Robotiq
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose
import numpy as np
from typing import Any, Dict, Union
import torch

from rh20tp_envs.base_env import Rh20tBaseEnv
from rh20tp_envs.loaders import OBJECTS_REGISTRY, TABLES_REGISTRY
import rh20tp_envs.strategies as strategy


@register_env("Rh20t-PickObject-v0", max_episode_steps=500)
class Rh20tPickObjectEnv(Rh20tBaseEnv):
    SUPPORTED_OBJECTS = [
        "wooden_cube",
        "large_wooden_cube",
        "standing_coke_can",
        "fallen_coke_can",
        "screw_driver",
    ]
    INSTRUCTIONS = [
        "Pick the {object}.",
    ]
    lifted_goal_thresh = 0.15

    agent: Union[Panda, Xmate3Robotiq, Fetch]

    def __init__(self, *args, robot_uids="panda", robot_init_qpos_noise=0.02, **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.is_object_lifted = None
        super().__init__(*args, robot_uids=robot_uids, **kwargs)
    
    def get_language_instruction(self):
        instruction = np.random.choice(self.INSTRUCTIONS, (1,), replace=True).tolist()[0]
        if "{object}" in instruction:
            object_type = " ".join(self.object_type.split("_"))
            instruction = instruction.replace("{object}", object_type)
        return instruction

    def _load_scene(self, options: Dict):
        # Load random table scene
        table_type = np.random.choice(
            self.SUPPORTED_TABLES, (1,), replace=True).tolist()[0]
        self.table_scene = TABLES_REGISTRY[table_type](
            self, robot_init_qpos_noise=self.robot_init_qpos_noise)

        # Load random object into the scene
        self.object_type = np.random.choice(
            self.SUPPORTED_OBJECTS, (1,), replace=True).tolist()[0]
        self.object, self.object_height, self.object_hooks = \
            OBJECTS_REGISTRY[self.object_type](self.scene)

    def _initialize_episode(self, env_idx: torch.Tensor, options: Dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            # Initialize object position and posture
            xyz = torch.zeros((b, 3))
            xyz[:, 0] = torch.rand((b,)) * 0.4 - 0.3
            xyz[:, 1] = torch.rand((b,)) * 0.5 - 0.45
            xyz[:, 2] = self.object_height
            qs = randomization.random_quaternions(b, lock_x=True, lock_y=True)
            self.object.set_pose(Pose.create_from_pq(xyz, qs))

            if 'init' in self.object_hooks:
                self.object_hooks['init'](self.object)

    def _before_simulation_step(self):
        if 'before_simulation' in self.object_hooks:
            self.object_hooks['before_simulation'](self.object)

    def _get_obs_extra(self, info: Dict):
        obs = dict(
            is_object_grasped=info["is_object_grasped"],
            tcp_pose=self.agent.tcp.pose.raw_pose,
            goal_pos=self.object.pose.raw_pose,
        )
        if "state" in self.obs_mode:
            obs.update(
                obj_pose=self.object.pose.raw_pose,
                tcp_to_obj_pos=self.object.pose.p - self.agent.tcp.pose.p,
            )
        return obs

    def evaluate(self):
        with torch.device(self.device):
            b = self.object.pose.shape[0]
            if self.is_object_lifted is None:
                self.is_object_lifted = torch.zeros((b,), dtype=torch.bool)

            is_object_grasped = self.agent.is_grasping(self.object)
        
            lifted_height = (self.object.pose.p[:, 2] - self.object_height).clip(min=0)
            is_object_lifted = lifted_height >= self.lifted_goal_thresh
            self.is_object_lifted = self.is_object_lifted | is_object_lifted

        return {
            "success": self.is_object_lifted,
            "is_object_grasped": is_object_grasped,
            "is_object_lifted": self.is_object_lifted,
            "lifted_height": lifted_height,
        }

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        tcp_to_obj_dist = torch.linalg.norm(
            self.object.pose.p - self.agent.tcp.pose.p, axis=1
        )
        reaching_reward = 1 - torch.tanh(5 * tcp_to_obj_dist)
        reward = reaching_reward

        is_object_grasped = info["is_object_grasped"]
        reward += is_object_grasped

        lifted_height = info["lifted_height"]
        lifted_height = (lifted_height / self.lifted_goal_thresh).clip(0, 1)
        reward += lifted_height

        reward[info["success"]] = 3
        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 3
    
    def single_gt_action_space(self, idx):
        if self.action_time_step < 30:
            return strategy.wait(self.action_time_step)
        
        agent_tcp = self.agent.tcp.pose.raw_pose[idx]
        object_pose = self.object.pose.raw_pose[idx]

        if self.is_object_lifted[idx]:
            # import ipdb; ipdb.set_trace()
            self.success_time_step[idx] = self.success_time_step[idx] + 1
            if self.success_time_step[idx] < 20:
                return torch.tensor([0,0,0.1,0,0,0,0],dtype=torch.float32)
            return torch.tensor([0,0,0,0,0,0,0],dtype=torch.float32)
        
        is_grasped = self.agent.is_grasping(self.object)[idx]
        if is_grasped:
            self.last_grasp_time[idx] = 5
            return strategy.up(is_grasped)
        else:
            return strategy.pick(agent_tcp, object_pose, self.last_grasp_time[idx])

    def gt_action_space(self):
        b = self.agent.tcp.pose.raw_pose.shape[0]

        self.action_time_step = self.action_time_step + 1 if hasattr(self, "action_time_step") else 0
        if hasattr(self, "last_grasp_time"):
            self.last_grasp_time = [time - 1 for time in self.last_grasp_time]
        else:
            self.last_grasp_time = [0] * b
        if not hasattr(self, 'success_time_step'):
            self.success_time_step = [0] * b

        with torch.device(self.device):
            action = torch.zeros((b, 7), dtype=torch.float32)

            for idx in range(b):
                action_i = self.single_gt_action_space(idx)
                action[idx] = action_i

            if b == 1:
                action = action.squeeze(0)

        return action
