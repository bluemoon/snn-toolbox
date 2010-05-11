include "conf.pxi"

import random

import  numpy as np
cimport numpy as np

import  base
cimport base

cdef class neurons(base.neurons_base):
    #def __init__(self, neurons):
    #    base.neurons_base.__init__(neurons)
    pass
    
cdef class layer(base.layer_base):
    def __init__(self, previous_neurons, next_neurons):
        shape = (previous_neurons.size, next_neurons.size, SYNAPSES)
        base.layer_base.__init__(self, previous_neurons, next_neurons, shape)
         
    cdef void forward_implementation(self):
        pass
    
    cdef void backward_implementation(self):
        pass
    
    cdef void forward(self):
        self.forward_implementation()
        
    cdef void backward(self):
        self.backward_implementation()
    
    cdef void activate(self, np.ndarray input):
        #assert self.prev.time.shape == input.shape
        
        self.prev.time = input
        self.forward()
