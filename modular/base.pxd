cimport numpy as np
#import numpy as np


cdef class neurons_base:
    cdef readonly int size
    cdef public np.ndarray time
    cdef public np.ndarray desired_time
    
cdef class layer_base:
    cdef public object prev
    cdef public object next
    cdef public np.ndarray weights
    cdef public np.ndarray deltas
    cdef public np.ndarray derivative
    cdef public np.ndarray weight_delta
    cdef public double learning_rate
    cdef public int threshold
    cdef public int prev_dim, next_dim
    cdef public bint last_layer
    
    cdef str weight_method

    cpdef forward_implementation(self)
    cpdef backward_implementation(self)
    cpdef forward(self)
    cpdef backward(self)
    cpdef activate(self, np.ndarray)

cdef class network_base:
    cdef list layers
    cdef object layer, output_layer
    cdef object input_layer
    cdef int  layer_idx, layer_length
    cdef bint failed

    cpdef error(self)
    cpdef bint last_layer(self)
    cdef bint first_layer(self)
    cdef bint hidden_layer(self)
    cpdef forward_pass(self, np.ndarray, np.ndarray)