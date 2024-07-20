from mani_skill.agents.robots.fetch import FETCH_WHEELS_COLLISION_BIT
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils.structs.actor import Actor
import numpy as np
import sapien


def init_table_robot(
    b,
    env: BaseEnv, 
    ground: Actor, 
    robot_init_qpos_noise: float, 
    table_height: float
):
    if env.robot_uids == "panda":
        qpos = np.array([
                       0.0, np.pi / 8,           0.0,
            -np.pi * 5 / 8,       0.0, np.pi * 3 / 4,
                 np.pi / 4,      0.04,          0.04,
        ])
        qpos = env._episode_rng.normal(
            0, robot_init_qpos_noise, (b, len(qpos))) + qpos
        qpos[:, -2:] = 0.04
        env.agent.reset(qpos)
        env.agent.robot.set_pose(sapien.Pose([-0.615, 0, 0]))
    
    elif env.robot_uids == "panda_wristcam":
        qpos = np.array([
                       0.0, np.pi / 8,           0.0, 
            -np.pi * 5 / 8,       0.0, np.pi * 3 / 4, 
                -np.pi / 4,      0.04,          0.04,
        ])
        qpos = env._episode_rng.normal(
            0, robot_init_qpos_noise, (b, len(qpos))) + qpos
        qpos[:, -2:] = 0.04
        env.agent.reset(qpos)
        env.agent.robot.set_pose(sapien.Pose([-0.615, 0, 0]))

    elif env.robot_uids == "xmate3_robotiq":
        qpos = np.array([
                   0.0, np.pi / 6,       0.0, 
             np.pi / 3,       0.0, np.pi / 2, 
            -np.pi / 2,       0.0,       0.0,
        ])
        qpos = env._episode_rng.normal(
            0, robot_init_qpos_noise, (b, len(qpos))) + qpos
        qpos[:, -2:] = 0
        env.agent.reset(qpos)
        env.agent.robot.set_pose(sapien.Pose([-0.562, 0, 0]))

    elif env.robot_uids == "fetch":
        qpos = np.array([
            0, 0, 0, 0.386, 0,
            0, 0, -np.pi / 4, 0, np.pi / 4,
            0, np.pi / 3, 0, 0.015, 0.015,
        ])
        env.agent.reset(qpos)
        env.agent.robot.set_pose(
            sapien.Pose([-1.05, 0, -table_height]))
        ground.set_collision_group_bit(
            group=2, bit_idx=FETCH_WHEELS_COLLISION_BIT, bit=1)
    
    elif env.robot_uids == ("panda", "panda"):
        agent: MultiAgent = env.agent
        qpos = np.array([
                       0.0, np.pi / 8,           0.0,
            -np.pi * 5 / 8,       0.0, np.pi * 3 / 4,
                 np.pi / 4,      0.04,          0.04,
        ])
        qpos = env._episode_rng.normal(
            0, robot_init_qpos_noise, (b, len(qpos))) + qpos
        qpos[:, -2:] = 0.04
        
        agent.agents[1].reset(qpos)
        agent.agents[1].robot.set_pose(
            sapien.Pose([0, 0.75, 0], q=euler2quat(0, 0, -np.pi / 2)))
        agent.agents[0].reset(qpos)
        agent.agents[0].robot.set_pose(
            sapien.Pose([0, -0.75, 0], q=euler2quat(0, 0, np.pi / 2)))

    elif "dclaw" in self.env.robot_uids or \
          "allegro" in self.env.robot_uids or \
          "trifinger" in self.env.robot_uids:
        # Need to specify the robot qpos for each sub-scenes using tensor api
        pass
