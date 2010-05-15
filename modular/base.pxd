cimport numpy as np
#import numpy as np
ctypedef struct neuron_t:
    double *time
    np.PyArrayObject *time_arr
    np.PyArrayObject *desired_time
    int size

cdef class neurons_base:
    #cdef neuron_t neuron
    #cdef double *time_data
    cdef readonly int size
    cdef public:
        np.ndarray time
        np.ndarray desired_time

    
cdef class layer_base:
    cdef public:
        neurons_base prev, next
        np.ndarray weights
        np.ndarray deltas
        np.ndarray derivative
        np.ndarray weight_delta
        double learning_rate
        int threshold
        int prev_dim, next_dim
        bint last_layer

    
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
