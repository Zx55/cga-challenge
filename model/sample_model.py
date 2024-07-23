import math
import torch
import numpy as np
from model_wrapper import ModelWrapper

class SampleModel(ModelWrapper):
    def __init__(self) -> None:
        super().__init__()
    
    def load_model(self):
        self.model = None
        self.policy = None
    
    def initialize(self):
        self.model = None
        self.policy = None
        self.success_time_step = 0

    def pred_action(self, input_data):
        out_numpy = self.gt_action_space(**input_data)
        out_tensor = torch.from_numpy(out_numpy).cuda()
        return out_tensor.detach().cpu().numpy()
    
    def gt_action_space(self, image, agent_tcp, **kwargs):
        return np.array([0,0,0,0,0,0,-1],dtype=np.float32)