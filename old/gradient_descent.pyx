import numpy  as np
cimport numpy as np

class GradientDescent:
    def __init__(self):
        # --- BackProp parameters ---
        # learning rate (0.1-0.001, down to 1e-7 for RNNs)
        self.alpha = 0.1
        
        # alpha decay (0.999; 1.0 = disabled)
        self.alphadecay = 1.0
    
        # momentum parameters (0.1 or 0.9)
        self.momentum = 0.0
        self.momentumvector = None
 
        # --- RProp parameters ---
        self.rprop = False
        # maximum step width (1 - 20)
        self.deltamax = 5.0
        # minimum step width (0.01 - 1e-6)
        self.deltamin = 0.01
        # the remaining parameters do not normally need to be changed
        self.deltanull = 0.1
        self.etaplus = 1.2
        self.etaminus = 0.5
        self.lastgradient = None
        
    def init(self, values):
        assert isinstance(values, np.ndarray)
        self.values = values.copy()
        if self.rprop:
            self.lastgradient = np.zeros(len(values), dtype='float64')
            self.rprop_theta = self.lastgradient + self.deltanull      
            self.momentumvector = None
        else:
            self.lastgradient = None
            self.momentumvector = np.zeros(len(values))
