cimport numpy as np
import  numpy as np
"""
cdef class layer:
    cdef public np.ndarray weights
    cdef public np.ndarray deltas
    cdef public np.ndarray derivative
    cdef public np.ndarray derive_delta
    cdef public object prev
    cdef public object next
    cdef object weight_method
    cdef double _learning_rate
"""
    
cdef class neurons:
    cdef public np.ndarray time
    cdef public np.ndarray desired_time
    cdef int neurons
    cpdef inline int size(self)
