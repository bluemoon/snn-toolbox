# encoding: utf-8
# cython: profile=False
# cython: boundscheck=False
# cython: wraparound=False
# cython: infer_types=True
include "conf.pxi"
cimport numpy as np
import  numpy as np
import  random

cdef class neurons_base:
    def __init__(self, neurons):
        self.size         = neurons
        self.time         = np.ndarray((neurons))
        self.desired_time = np.ndarray((neurons))

    #cpdef inline int size(self):
    #    return self.neurons
    
cdef class layer_base:
    def __init__(self, previous_neurons, next_neurons, shape): 
        previous = previous_neurons.size
        next     = next_neurons.size

        self.prev  = previous_neurons
        self.next  = next_neurons
        self.weights = np.random.rand(*shape) * 10.0
        self.deltas  = np.ndarray((next))
        self.derivative   = np.random.rand(*shape)
        self.weight_delta = np.random.rand(*shape)
        self.learning_rate = 1.0

        self.weight_method = 'random2'
        
        if self.weight_method == 'random1':
            for i in xrange(self.prev.size):
                for h in xrange(self.next.size):
                    r = random.randint(1, 10) 
                    for k in xrange(SYNAPSES):
                        self.weights[i, h, k] = r

        elif self.weight_method == 'random2':
            for i in xrange(self.prev.size):
                for h in xrange(self.next.size):
                    for k in xrange(SYNAPSES):
                        r = random.randint(1, 10)
                        self.weights[i, h, k] = r

        elif self.weight_method == 'normalized':
            mu, sigma = 1, 0.11
            self.weights = np.random.normal(mu, sigma, size=(previous, next, SYNAPSES))


cdef class network_base:
    def __init__(self, layers):
        self.layers = layers
        
