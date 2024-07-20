from mani_skill.envs.sapien_env import BaseEnv
import os

from rh20tp_envs.loaders.tables.rh20t_table import Rh20tTableSceneBuilder


class Rh20tTableWithGreenClothV1SceneBuilder(Rh20tTableSceneBuilder):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.assets_dir = os.path.join(os.path.dirname(__file__), "assets")


def build_rh20t_table_green_cloth_v1_scene(env: BaseEnv, robot_init_qpos_noise: float):
    table_scene = Rh20tTableWithGreenClothV1SceneBuilder(env, robot_init_qpos_noise)
    table_scene.build()
    return table_scene
