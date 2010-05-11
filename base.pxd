cimport numpy as np
import numpy as np

cdef class neurons_base:
    cdef public int size
    cdef public np.ndarray time, desired_time
    
cdef class layer_base:
    cdef public object prev, next
    cdef public np.ndarray weights, deltas, derivative, weight_delta
    cdef public double learning_rate
    cdef str weight_method
cdef class network_base:
    cdef layer_base layer
    cdef object layers

