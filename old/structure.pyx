include "conf.pxi"
import numpy as np
cimport numpy as np
import random


cdef class neurons:
    def __init__(self, neurons):
        self.neurons      = neurons
        self.time         = np.ndarray((neurons))
        self.desired_time = np.ndarray((neurons))

    cpdef inline int size(self):
        return self.neurons
    
class layer:
    def __init__(self, previous_neurons, next_neurons):
        previous = previous_neurons.size()
        next     = next_neurons.size()

        self.prev  = previous_neurons
        self.next  = next_neurons

        self.weights = np.random.rand(previous, next, SYNAPSES) * 10.0
        self.deltas  = np.ndarray((next))
        self.derivative   = np.random.rand(previous, next, SYNAPSES)
        self.derive_delta = np.random.rand(previous, next, SYNAPSES)

        self._learning_rate = 1.0

        self.weight_method = 'random2'
        
        if self.weight_method == 'random1':
            for i in xrange(self.prev.size()):
                for h in xrange(self.next.size()):
                    r = random.randint(1, 10) 
                    for k in xrange(SYNAPSES):
                        self.weights[i, h, k] = r

        elif self.weight_method == 'random2':
            for i in xrange(self.prev.size()):
                for h in xrange(self.next.size()):
                    for k in xrange(SYNAPSES):
                        r = random.randint(1, 255)
                        self.weights[i, h, k] = r

        elif self.weight_method == 'normalized':
            mu, sigma = 1, 0.11
            self.weights = np.random.normal(mu, sigma, size=(previous, next, SYNAPSES))

    @property
    def learning_rate(self):
        return self._learning_rate
