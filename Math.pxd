cimport numpy  as np
import numpy as np

cdef extern from "math.h" nogil:
    double c_exp "exp" (double)
    double c_modf "modf" (double, double*)
    double c_tan  "tan" (double)

cdef extern from "spike_prop.h" nogil:
    double c_e "e"(double)
    
cdef extern from "stdlib.h" nogil:
    int    c_rand  "rand" ()
    int    c_srand "srand" (int)
    double c_fmod  "fmod" (double, double)
    


cdef class Math:
    cdef object layers
    cdef object output_layer
    cdef object input_layer
    cdef object propagating_type
    cdef object propagating_routine
    cdef int threshold
    cdef bint failed
    cdef object layer
    cdef int layer_idx
    
    cpdef e(self, double)
    cpdef sign(self, int)
    cpdef spike_response_derivative(self, double)
    cpdef y(self, double, double, double)
    cpdef double excitation(self, np.ndarray, double, double)
    cpdef excitation_derivative(self, np.ndarray, double, double)
    cpdef error_weight_derivative(self, double, double, double, double)
    cpdef change(self, double, double, double, double)
    cpdef delta_j(self, int)
    cpdef delta_i(self, int)
    cpdef delta_j_bottom(self, int)
    cpdef delta_i_bottom(self, int)
    cpdef delta_i_top(self, int)
    
