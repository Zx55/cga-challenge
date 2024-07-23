import cv2
from datetime import datetime
import gymnasium as gym
import mani_skill.envs
from mani_skill.utils.wrappers import RecordEpisode
import os
import pytz

from rh20tp_envs import *
from utils import init_seed, list_envs, get_args, Controller

os.environ['GIT_PYTHON_REFRESH'] = 'quiet'

def main(args):
    current_time = datetime.now(pytz.timezone('Asia/Shanghai')).strftime("%Y%m%d_%H%M%S")
    work_dir = os.path.join('results', current_time)
    os.makedirs(work_dir, exist_ok=True)
    for i in range(args.num_envs):
        os.makedirs(os.path.join(work_dir, f'env_{i}'), exist_ok=True)

    init_seed(args.seed)

    env = gym.make(
        args.env,
        num_envs=args.num_envs,
        obs_mode="state", # there is also "state_dict", "rgbd", ...
        control_mode="pd_ee_delta_pose", # there is also "pd_joint_delta_pos", ...
        render_mode="rgb_array" # "rgb_array" or "human"
    )
    instruction = env.get_language_instruction()
    max_steps = env._saved_kwargs['max_episode_steps']
    print("Observation space:", env.observation_space)
    print("Action space:", env.action_space)
    print("TaskName:", args.env)
    print("Instruction:", instruction)

    if args.gt is True:
        if hasattr(env, "gt_action_space"):
            print('sample gt trajectory')
        else:
            print(f'warning: env {args.env} has no gt action space')
    else:
        print('sample random actions')

    env = RecordEpisode(
        env, 
        os.path.join(work_dir, args.env),
        info_on_video=args.num_envs == 1,
        max_steps_per_video=max_steps)
    obs, _ = env.reset(seed=args.seed) # reset with a seed for determinism

    controller = Controller(env=env, work_dir=work_dir, args=args)
    controller.start_service()

if __name__ == '__main__':
    args = get_args()
    
    if args.list_envs:
        list_envs(verbose=True)
        exit(0)

    main(args)
