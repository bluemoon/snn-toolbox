DEF WEAVE       = False
DEF MP          = False
DEF DECAY       = 7
DEF SYNAPSES    = 16
DEF IPSP        = 1
DEF MAX_TIME    = 50
DEF TIME_STEP   = 0.01
DEF NEG_WEIGHTS = False

DEF QUICKPROP   = False
DEF RPROP       = False

cdef extern from "spike_prop.h" nogil:
    double c_e "e"(double)


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
    
import  numpy  as np
cimport numpy  as np
cimport python as py
