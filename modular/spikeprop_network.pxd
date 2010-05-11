import  numpy as np
cimport numpy as np
import  base
cimport base

## network, module, connections
cdef class modular2(base.network_base):
    cpdef forward_pass(self, np.ndarray, np.ndarray):
