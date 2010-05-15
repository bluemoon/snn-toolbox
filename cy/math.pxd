include "../misc/conf.pxi"
cimport modular.spike_types as types

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
    #cdef types.layer *layers
    cdef:
        list layers
        types.layer output_layer
        types.layer input_layer
        types.layer layer
        object propagating_type
        int threshold, layer_idx
        int layer_length
        bint failed

        
    cdef double e(self, double)
    cdef int sign(self, int)
    cdef double spike_response_derivative(self, double)
    cdef double y(self, double, double, double)
    cdef inline double excitation(self, double *, double, double)
    cdef double excitation_derivative(self, double *, double, double)
    cdef double error_weight_derivative(self, double, double, double, double)
    cdef double change(self, double, double, double, double)
    cdef delta_j(self, int)
    cdef delta_i(self, int)
    cdef delta_j_bottom(self, int)
    cdef delta_i_bottom(self, int)
    cdef delta_i_top(self, int)
    
