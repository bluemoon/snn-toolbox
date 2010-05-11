include "conf.pxi"

import  numpy as np
cimport numpy as np
import  base
cimport base

## network, module, connections
cdef class modular2(base.network_base):
    def __init__(self, layers):
        base.network_base.__init__(self, layers)
        
    cpdef forward_pass(self, np.ndarray input_times, np.ndarray desired_times):
        ## The first layer will be the furthest most left
        ## and the last layer will be the furthest most right
        ## 0 and -1 respectively
        self.input_layer.prev.time           = input_times
        self.output_layer.next.desired_time  = desired_times
        cdef int h, i, k, z, delay, layer_idx
        cdef double time  = 0, total = 0, ot = 0
        cdef double spike_time, t2, weight, q
        cdef int next_size, prev_size
        cdef double *prev_time, *next_time, *weights
        cdef np.ndarray prev_array, next_array, layer_weights
        
        for layer_idx in xrange(self.layer_length):
            self.layer = self.layers[layer_idx]
            self.layer_idx = layer_idx
            prev = self.layer.prev
            next = self.layer.next
            prev_size = prev.size
            next_size = next.size

            prev_array = <np.ndarray>prev.time
            next_array = <np.ndarray>next.time
            
            prev_time = <double *>prev_array.data
            next_time = <double *>next_array.data
            
            ## we now need to run from layer to layer
            ## first we must go through the next layer size(i)
            ## then we go through the previous layer(h)
            ## get the firing time of the previous layer(h)
            ## and figure out if the sum of all in our current time(time)
            ## passes the threshold value which is calculated with 
            ## spikeprop_math.linkout but because we are a subclass of it
            ## it can be accessed through self
            layer_weights = <np.ndarray>self.layer.weights
            for i in range(next_size):
                total = 0
                time  = 0
                while (total < self.threshold and time < MAX_TIME):
                    total = 0
                    for h in range(prev_size):
                        spike_time = prev_time[h]
                        ## when the time is past the spike time
                        if time >= spike_time:
                            weights = <double *>np.PyArray_GETPTR2(layer_weights, h, i)
                            z = int(time-spike_time)
                            ot = 0.0
                            for k from 0 <= k < z:
                                delay = k + 1
                                weight = weights[k]
                                ot += (weight * c_e(time-spike_time-delay))
                                
                            if self.last_layer():
                                if h >= (prev_size - IPSP):
                                    total -= ot
                                else:
                                    total += ot
                            else:
                                total += ot
                                
                    ## now set the next layers spike time to the current time
                    ## XXX: check to see if this can be optimized    
                    time += TIME_STEP
                    
                next_time[i] = time-TIME_STEP
                
                if time >= 50.0:
                    self.failed = True
                    break

    
