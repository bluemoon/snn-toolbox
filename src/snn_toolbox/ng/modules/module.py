import numpy as np

class module:
    paramdim = 0
    sequential = False
    bufferlist = None
    
    def __init__(self, in_dim, out_dim, name=None):
        self.indim = in_dim
        self.outdim = out_dim
        self.connection_head = None
        self.connection_tail = None
        
        self.time          = np.ndarray((neurons))
        self.desired_time  = np.ndarray((neurons))

        self.name = name
        self._resetBuffers()

    def _allBuffers(self):
        b = []
        for buffername, dim in self.bufferlist.items():
            b.append(getattr(self, buffername))
        return b
            
    def _resetBuffers(self, length=0):
        for buffername, dim in self.bufferlist.items():
            setattr(self, buffername, np.zeros(dim))
            
    def forward(self):
        self._forwardImplementation(self.inputbuffer, self.outputbuffer)
        
    def backward(self):
        self._backwardImplementation(self.outputerror, self.inputerror, self.outputbuffer, self.inputbuffer)
