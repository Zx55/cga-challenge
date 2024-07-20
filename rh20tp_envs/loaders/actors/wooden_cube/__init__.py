import os

from rh20tp_envs.loaders.actors.builder import actor_loader


def build_wooden_cube(scene, name: str = 'wooden_cube'):
    asset_dir = os.path.join(os.path.dirname(__file__), 'assets')
    wooden_cube = actor_loader(scene, asset_dir, name, density=200.0)
    wooden_cube_height = 0.025
    return wooden_cube, wooden_cube_height, {}


def build_large_wooden_cube(scene, name: str = 'wooden_cube'):
    asset_dir = os.path.join(os.path.dirname(__file__), 'assets')
    wooden_cube = actor_loader(scene, asset_dir, name, scale=1.25, density=200.0)
    wooden_cube_height = 0.025 * 1.35
    return wooden_cube, wooden_cube_height, {}
