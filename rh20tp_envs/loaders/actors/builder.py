from mani_skill.envs.scene import ManiSkillScene
import os
import sapien


def actor_loader(
    scene: ManiSkillScene,
    asset_dir: str,
    name: str,
    scale: float = 1.0,
    pose: sapien.Pose = sapien.Pose(),
    physical_material = None,
    density: float = 1000.0,
):
    if physical_material is None:
        # ideal rigid body
        physical_material = scene.create_physical_material(
            static_friction=0.5,
            dynamic_friction=0.5,
            restitution=0.0) 
    if isinstance(scale, float):
        scale = [scale] * 3

    builder = scene.create_actor_builder()

    # add collision
    decomposition = 'none'
    if os.path.exists(os.path.join(asset_dir, 'collision.obj.coacd.ply')):
        collision_file = os.path.join(asset_dir, 'collision.ply')
        os.rename(os.path.join(asset_dir, 'collision.obj.coacd.ply'), collision_file)
    elif os.path.exists(os.path.join(asset_dir, 'collision.ply')):
        collision_file = os.path.join(asset_dir, 'collision.ply')
    else:
        collision_file = os.path.join(asset_dir, 'collision.obj')
        decomposition = 'coacd'
    builder.add_multiple_convex_collisions_from_file(
        filename=collision_file,
        scale=scale,
        material=physical_material,
        density=density,
        pose=pose,
        decomposition=decomposition)
    
    # add visual file
    if os.path.exists(os.path.join(asset_dir, 'textured.dae')):
        visual_file = os.path.join(asset_dir, 'textured.dae')
    elif os.path.exists(os.path.join(asset_dir, 'textured.glb')):
        visual_file = os.path.join(asset_dir, 'textured.glb')
    else:
        visual_file = os.path.join(asset_dir, 'textured.obj')
    builder.add_visual_from_file(
        filename=visual_file, 
        scale=scale,
        pose=pose)
    
    return builder.build(name=name)
