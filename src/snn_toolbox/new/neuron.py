import numpy as np

class neurons:
    def __init__(self, size):
        self.size = size
        self.time = np.ndarray(size)
        self.desired_time = np.ndarray(size)
    def __repr__(self):
        return '<neuron size: %d>' % self.size
