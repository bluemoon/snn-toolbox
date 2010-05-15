DEF WEAVE       = False
DEF MP          = False
DEF DECAY       = 7
DEF SYNAPSES    = 16
DEF IPSP        = 1
DEF MAX_TIME    = 50
DEF TIME_STEP   = 0.1
DEF NEG_WEIGHTS = False

DEF QUICKPROP   = False
DEF RPROP       = False

cdef extern from "spike_prop.h" nogil:
    double c_e "e"(double)

cdef extern from "stdlib.h" nogil:
    int c_rand "rand"()
    
import  numpy  as np
cimport numpy  as np
cimport python as py
