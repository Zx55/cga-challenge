import numpy as np
import random
import torch


def init_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def list_envs(verbose=False):
    from mani_skill.utils.registration import REGISTERED_ENVS
    envs_name = [env for env in REGISTERED_ENVS.keys() if env.startswith('Rh20t')]
    
    if not verbose:
        return envs_name
    
    print('Rh20t Supported Environments:')
    for env_name in envs_name:
        print(f'- {env_name}')
    print(f'Totally {len(envs_name)} Environments.')
