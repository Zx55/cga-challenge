from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils.scene_builder.table import TableSceneBuilder


def build_table_default_scene(env: BaseEnv, robot_init_qpos_noise: float):
    table_scene = TableSceneBuilder(env, robot_init_qpos_noise)
    table_scene.build()
    return table_scene
