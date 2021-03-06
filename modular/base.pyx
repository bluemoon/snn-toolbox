# encoding: utf-8
# cython: profile=False
# cython: boundscheck=False
# cython: wraparound=False
# cython: infer_types=True

include "../misc/conf.pxi"

cimport numpy as np
import  numpy as np
import  random


cdef class neurons_base:
    def __init__(self, int neurons):
        self.size          = neurons
        self.time          = np.ndarray((neurons))
        self.desired_time  = np.ndarray((neurons))
        #self.time_data     = <double *>self.time.data

    def __repr__(self):
        return '<neuron size:%d>' % self.size
    
cdef class layer_base:
    def __init__(self, neurons_base previous_neurons, neurons_base next_neurons, shape): 
        cdef int previous = previous_neurons.size
        cdef int next     = next_neurons.size

        self.prev  = previous_neurons
        self.next  = next_neurons

        self.prev_dim = previous
        self.next_dim = next
        
        self.weights = np.ndarray(shape)
        self.deltas  = np.ndarray((next))
        self.derivative   = np.ndarray(shape) 
        self.weight_delta = np.ndarray(shape) 
        self.learning_rate = 1.0
        self.threshold = 50
        self.weight_method = 'random1'
        
        if self.weight_method == 'random1':
            for i in xrange(self.prev.size):
                for h in xrange(self.next.size):
                    r = random.randint(1, 10)
                    #r = c_rand() % 10.0
                    for k in xrange(SYNAPSES):
                        self.weights[i, h, k] = r

        elif self.weight_method == 'random2':
            for i in xrange(self.prev.size):
                for h in xrange(self.next.size):
                    for k in xrange(SYNAPSES):
                        r = random.randint(1, 10)
                        self.weights[i, h, k] = r

        elif self.weight_method == 'normalized':
            mu, sigma = 1, 3
            self.weights = np.random.normal(mu, sigma, size=(previous, next, SYNAPSES))

    def __repr__(self):
        return '<layer prev size:%d next size: %d>' % (self.prev.size, self.next.size)

    cpdef forward_implementation(self):
        pass
    
    cpdef backward_implementation(self):
        pass
     
    cpdef forward(self):
        self.forward_implementation()
        
    cpdef backward(self):
        self.backward_implementation()
    
    cpdef activate(self, np.ndarray input):
        pass 
    
    
cdef class network_base:
    def __init__(self, layers):
        self.layers = layers
        self.layer_length = len(self.layers)
        self.input_layer  = layers[0]
        self.output_layer = layers[-1] 
        self.failed = False
        self.layer = None
        
    cpdef bint last_layer(self):
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
            
    cpdef error(self):
        cdef int j
        cdef object last_layer = self.layers[-1]
        cdef double total = 0.0
        for j in range(last_layer.next.size):
            total += (last_layer.next.time[j] - last_layer.next.desired_time[j]) ** 2.0
            
        return (total/2.0)

    cpdef forward_pass(self, np.ndarray input, np.ndarray desired):
        pass



    
