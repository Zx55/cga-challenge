import os
import cv2
import time
import numpy as np

from mani_skill.utils.wrappers import RecordEpisode

class Controller:
    def __init__(self, env: RecordEpisode, work_dir, args=None, communication_dir='comm') -> None:
        self.communication_dir = communication_dir
        os.makedirs(self.communication_dir, exist_ok=True)
        self.input_file = os.path.join(self.communication_dir, 'input.npy')
        self.output_file = os.path.join(self.communication_dir, 'output.npy')
        self.env = env
        self.done = False
        self.work_dir = work_dir
        self.gt = args.gt
        self.num_env = args.num_envs
        self.timestamp = 0
        cam = self.env.scene.human_render_cameras['render_camera']
        cam_param = cam.get_params()
        for key in cam_param.keys():
            cam_param[key] = cam_param[key].numpy()
        self.camera = cam_param
        self.instruction = self.env.get_language_instruction()

    def kill_signal(self):
        return self.done
    
    def get_observation(self):
        image = self.env.render().cpu().numpy()
        assert image.shape[0] == self.num_env
        for env_i in range(self.num_env):
            image_file = os.path.join(self.work_dir, f'env_{env_i}', f'step_{self.timestamp}.png')
            cv2.imwrite(image_file, image[env_i][:, :, ::-1])
        res_dict = self.camera
        res_dict.update(dict(
            image = image[:, :, :, ::-1], # num_env, 512, 512, 3
            agent_tcp = self.env.agent.tcp.pose.raw_pose[0].numpy(), # tensor size=[7]
            instruction = str(self.instruction)
        ))
        return res_dict 
    
    def execute(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.done = truncated
        self.timestamp += 1
        if self.timestamp >= self.env.max_steps_per_video:
            self.done = True

    def start_service(self):
        while not self.kill_signal():
            input_data = self.get_observation()
            np.save(self.input_file, input_data)
            
            if self.gt and hasattr(self.env, "gt_action_space"):
                action = self.env.gt_action_space()
            else:
                while True:
                    if os.path.exists(self.output_file):
                        time.sleep(1)
                        action = np.load(self.output_file, allow_pickle=True)
                        os.remove(self.output_file)
                        break

            self.execute(action)
        self.env.close()
        np.save(self.input_file, {})
