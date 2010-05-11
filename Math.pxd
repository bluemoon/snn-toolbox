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
    
    cdef double e(self, double)
    cdef int sign(self, int)
    cdef spike_response_derivative(self, double)
    cdef double y(self, double, double, double)
    cdef double excitation(self, double *, double, double)
    cdef excitation_derivative(self, np.ndarray, double, double)
    cdef double error_weight_derivative(self, double, double, double, double)
    cdef double change(self, double, double, double, double)
    cdef delta_j(self, int)
    cdef delta_i(self, int)
    cdef delta_j_bottom(self, int)
    cdef delta_i_bottom(self, int)
    cdef delta_i_top(self, int)
    
