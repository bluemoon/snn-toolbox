# encoding: utf-8
# cython: profile=False
# cython: boundscheck=True
# cython: wraparound=False
# cython: infer_types=True
include "misc/conf.pxi"

import  numpy as np
cimport numpy as np
import  base
cimport base

## network, module, connections
cdef class modular2(base.network_base):
    def __init__(self, layers):
        base.network_base.__init__(self, layers)
        
    cpdef forward_pass(self, np.ndarray input, np.ndarray desired):
        self.layers[0].prev.time  = input
        self.layers[-1].next.time = desired
        for idx from 0 <= idx < self.layer_length:
            self.layer = self.layers[idx]
            prev = self.layer.prev
            next = self.layer.next 

            prev_size = prev.size
            next_size = next.size
            self.layer.last_layer = self.last_layer()
            self.layer.forward_implementation()
