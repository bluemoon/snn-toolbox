import copencl as cl
import clyther
from ctypes import c_float
from clyther import runtime as clrt
from clyther.memory import global_array_type
import numpy as np


@clyther.kernel
@clyther.bind('global_work_size' ,'weights.size')
@clyther.bind('local_work_size' , 1)
def excitation(weights, spike, time, ret):
    gid = clrt.get_global_id(0)
    loc = weights[gid]
    
