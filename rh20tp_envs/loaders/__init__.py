from .actors import *
from .articulations import *
from .tables import *


OBJECTS_REGISTRY = {
    'large_wooden_cube': build_large_wooden_cube,
    'wooden_cube': build_wooden_cube,
}

TABLES_REGISTRY = {
    'table_default': build_table_default_scene,
    'table_green_cloth_v1': build_rh20t_table_green_cloth_v1_scene,
    'table_green_cloth_v2': build_rh20t_table_green_cloth_v2_scene,
}
