include "conf.pxi"

import numpy as np
cimport numpy as np
import random

cimport base
import  base

cdef class neurons(base.neurons_base):
    #def __init__(self, neurons):
    #    base.neurons_base.__init__(neurons)
    pass
    
cdef class layer(base.layer_base):
    def __init__(self, previous_neurons, next_neurons):
        shape = (previous_neurons.size, next_neurons.size, SYNAPSES)
        base.layer_base.__init__(self, previous_neurons, next_neurons, shape)
