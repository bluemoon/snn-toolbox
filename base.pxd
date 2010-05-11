cimport numpy as np
import numpy as np

cdef class neurons_base:
    cdef public int size
    cdef public np.ndarray time, desired_time
    
cdef class layer_base:
    cdef public object prev
    cdef public object next
    cdef public np.ndarray weights
    cdef public np.ndarray deltas
    cdef public np.ndarray derivative
    cdef public np.ndarray weight_delta
    cdef public double learning_rate
    
    cdef str weight_method
    cdef int prev_dim, next_dim
    
    cdef void forward_implementation(self)
    cdef void backward_implementation(self)
    
    cdef void forward(self)
    cdef void backward(self)
    
    cdef void activate(self, np.ndarray)

cdef class network_base:
    cdef object layers, layer, output_layer
    cdef object input_layer, propagating_routine
    cdef int threshold, layer_idx, layer_length
    cdef bint failed

    cdef forward_pass(self, np.ndarray, np.ndarray)
    cdef double error(self)
    cdef inline bint last_layer(self)
    cdef bint first_layer(self)
    cdef bint hidden_layer(self)
