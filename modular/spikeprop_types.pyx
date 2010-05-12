# cython: profile=False
# cython: boundscheck=True
# cython: wraparound=False
# cython: infer_types=False
include "../misc/conf.pxi"
import sys, os
import random

import  numpy as np
cimport numpy as np

from base cimport *
from old.Math cimport *

cdef class neurons(neurons_base):
    pass
    
cdef class layer(layer_base):
    cdef object math
    def __init__(self, previous_neurons, next_neurons):
        shape = (previous_neurons.size, next_neurons.size, SYNAPSES)
        layer_base.__init__(self, previous_neurons, next_neurons, shape)
        self.math = Math()
        
    cpdef forward_implementation(self): 
        cdef int h, i, j
        cdef double *weights
        cdef double time, total, spike_time, ot
         
        for i in range(self.next.size):
            total = 0
            time  = 0
            while (total < self.threshold and time < MAX_TIME):
                total = 0
                for h in range(self.prev.size):
                    spike_time = self.prev.time[h]
                    if time >= spike_time:
                        weights = <double *>np.PyArray_GETPTR2(self.weights, h, i)
                        z = int(time-spike_time)
                        ot = 0

                        for k from 0 <= k < z:
                                delay = k + 1
                                weight = weights[k]
                                ot += (weight * c_e(time-spike_time-delay))

                        if self.last_layer:
                            if h >= (self.layer.prev.size - IPSP):
                                total -= ot
                            else:
                                total += ot
                        else:
                            total += ot
                            
                time += 0.01
                
            self.next.time[i] = (time - 0.01)
            if time >= 50.0:
                self.failed = True
                
        
    cpdef backward_implementation(self):
        pass
    
    cpdef forward(self):
        self.forward_implementation()
        
    cpdef backward(self):
        self.backward_implementation()
    
    cpdef activate(self, np.ndarray input):
        #assert self.prev.time.shape == input.shape
        
        self.prev.time = input
        self.forward()
