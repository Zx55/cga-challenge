import os
import time
import numpy as np

class ModelWrapper:
    def __init__(self) -> None:
        self.communication_dir = '/comm'
        self.input_file = os.path.join(self.communication_dir, 'input.npy')
        self.output_file = os.path.join(self.communication_dir, 'output.npy')
        self.action_time_step = 0
        self.load_model()
        self.initialize()
    
    def load_model(self):
        raise NotImplementedError

    def initialize(self):
        raise NotImplementedError

    def pred_action(self, input_data):
        '''
            dict(
                extrinsic_cv: np.array([1,3,4])
                cam2world_gl: np.array([1,4,4])
                instrinsic_cv: np.array([1,3,3])
                image: np.array([n,512,512,3])
                agent_tcp: np.array([7,])
            )
        
        '''
        raise NotImplementedError

    def kill_signal(self, input_data) -> bool:
        if len(input_data) == 0:
            return True
        return False
    
    def start_service(self):
        while True:
            if os.path.exists(self.input_file):
                time.sleep(1)
                input_data = np.load(self.input_file, allow_pickle=True).item()
                if not self.kill_signal(input_data):
                    self.action_time_step += 1
                    output_data = self.pred_action(input_data)
                    np.save(self.output_file, output_data)
                else:
                    self.action_time_step = 0 # init
                    self.initialize() 
                os.remove(self.input_file)
            