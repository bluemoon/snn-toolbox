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

cdef class layer_base:
    def __init__(self, previous_neurons, next_neurons, shape): 
        previous = previous_neurons.size
        next     = next_neurons.size

        self.prev  = previous_neurons
        self.next  = next_neurons

        self.prev_dim = previous
        self.next_dim = next
        
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
            
    cdef void forward_implementation(self):
        pass
    
    cdef void backward_implementation(self):
        pass
    
    cdef void forward(self):
        self.forward_implementation()
        
    cdef void backward(self):
        self.backward_implementation()
    
    cdef void activate(self, np.ndarray input):
        pass
    
    
cdef class network_base:
    def __init__(self, layers):
        self.layers = layers
        self.layer_length = len(self.layers)
        self.input_layer  = layers[0]
        self.output_layer = layers[-1]
        
    cdef inline bint last_layer(self):
        cdef bint last_layer
        if self.layer_idx == (self.layer_length - 1):
            last_layer = True
        else:
            last_layer = False

        return last_layer

    cdef bint first_layer(self):
        cdef bint first_layer
        if self.layer_idx == 0:
            first_layer = True
        else:
            first_layer = False
        return first_layer

    cdef bint hidden_layer(self):
        if not self.first_layer() and not self.last_layer():
            return True
        else:
            return False
            
    cdef double error(self):
        cdef int j
        cdef object last_layer = self.layers[-1]
        cdef double total = 0.0
        for j in range(last_layer.next.size):
            total += (last_layer.next.time[j] - last_layer.next.desired_time[j]) ** 2.0
            
        return (total/2.0)
        
    cdef forward_pass(self, np.ndarray input, np.ndarray desired):
        pass
