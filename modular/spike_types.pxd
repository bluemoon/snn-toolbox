# cython: profile=False
# cython: boundscheck=False
# cython: wraparound=False
# cython: infer_types=False
include "../misc/conf.pxi"
from base cimport *

cdef class neurons(neurons_base):
    pass
    
cdef class layer(layer_base):
    cdef:
        object math
