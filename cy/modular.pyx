# encoding: utf-8
# cython: profile=False
# cython: boundscheck=False
# cython: wraparound=False
# cython: infer_types=True
include "../misc/conf.pxi"

cimport numpy as np
import  numpy as np
cimport python as py

cimport cy.Math
import  cy.Math

#from modular.spikeprop_types  cimport  *
#from modular.spikeprop_types   import  *


IF WEAVE:
    from scipy.weave import *
    from numpy.distutils.system_info import get_info
    from os.path import join, split
IF MP:
    from multiprocessing import Process, Value, Array, Pool, Queue, Lock
    from corefunc import *
    import corefuncint
    
import os
import time as Time
import random
import numpy   as np
import cPickle as cp

## Checklist
##  [ ]Excitation


cdef class modular(cy.Math.Math):
    ## H (input), I (hidden) and J (output)
    ## 1) Firing time is x.j(excitation) ≥ v , progressing from the
    ##    hidden layer firing times, the firing times of
    ##    the output layer can be found, for a
    ##    given set of input firing times.
    ##
    ## 2) Calculate δj for all outputs according to (12)
    ##    For each subsequent layer I = J − {1...2}
    ##
    ## 3) Calculate δi for all neurons in I according to (17)
    ## 4) For output layer J , adapt w.ij by Δw.ij = −η * Y.i(t.j) * δj
    ##    For each subsequent layer I = J − {1...2}
    ## 6) Adapt w.hi by Δw.ij = −η * Y.h(t.i) * δi (18)
    ##
    ##   ∂E
    ## -------  = ε.ij^k(t-t.i-d.ij^k)δ.{i,j}
    ## ∂w.ij^k 
    ##
    cdef int layer_length
    cdef bint last
    
    def __init__(self, list layers):
        self.layers    = layers
        self.threshold = 50
        self.failed    = False
        self.layer     = None
        self.layer_idx = 0
        
        self.output_layer = self.layers[-1]
        self.input_layer  = self.layers[0]

        #self.propagating_type = 'descent'
        #self.propagating_routine = getattr(self, self.propagating_type + '_propagate')
        self.layer_length = len(self.layers)
        #self.propagating_routine = self.descent_propagate
        
    property fail:
        def __get__(self):
            return py.PyBool_FromLong(self.failed)
        
    cdef bint neg_weights(self):
        return NEG_WEIGHTS
    
    #property neg_weights:
    #    def __get__(self):
    #        return NEG_WEIGHTS
    #@cy.profile(False)
    cdef inline bint last_layer(self):
        cdef bint last_layer
        if self.layer_idx == (self.layer_length - 1):
            last_layer = True
        else:
            last_layer = False

        return last_layer

    cdef bint first_layer(self):
        cdef bint first_layer
        if self.layer_idx == 0:
            first_layer = True
        else:
            first_layer = False
        return first_layer

    cdef bint hidden_layer(self):
        if not self.first_layer() and not self.last_layer():
            return True
        else:
            return False
            
    cpdef double quickprop_propagate(self, int i, int j, int k, double actual, double spike, int delay, double delta):
    #cpdef double quickprop_propagate(self, sPropagate prop):

        cdef next_step = 0.0
        mu = 2.0
        shrink = mu /(1.0 + mu)
        decay = 0.0001
        
        weight = self.layer.weights[i,j,k]
        double_prime = self.layer.derivative[i,j,k]
        prime = self.error_weight_derivative(actual, 
                                             spike, 
                                             delay, 
                                             delta)

        d =  self.layer.derive_delta[i,j,k]
        p =  double_prime
        s =  prime + (decay * weight)

        self.layer.derive_delta[i, j, k] = next_step
        self.layer.derivative[i, j, k]   = prime
        return next_step

    cdef double descent_propagate(self, int i, int j, int k, double actual, double spike, int delay, double delta):
        ## 1) ncalls: 225280 tottime: 0.360 per: 0.000001598
        ## 2) ncalls: 304640 tottime: 0.489 per: 0.000001605
        cdef int layer, m
        cdef double delta_weight
        if not self.last_layer():
            layer = self.layer.next.size
            m = j
        else:
            layer = self.layer.prev.size
            m = i

        delta_weight = (-self.layer.learning_rate * c_e(actual - spike -delay) * delta)
        if m >= (layer - IPSP):
            #debug("IPSP Call: neuron: %d layer: %d layer size: %d" % (m, self.layer_idx, layer))
            #delta_weight = -self.change(actual, spike, delay, delta)
            delta_weight = -delta_weight

    
        return delta_weight

    cdef void _delta_j(self):
        cdef int j
        
        self.layer = self.layers[-1]
        self.layer_idx = self.layer_length-1
        for j in xrange(self.layer.next.size):
            self.layer.deltas[j] = self.delta_j(j)
            
    cdef void _delta_i(self):
        cdef int layer_idx
        for layer_idx in xrange(self.layer_length-1, -1, -1):
            self.layer_idx = layer_idx
            self.layer = self.layers[self.layer_idx]
            if not self.last_layer():
                for i in xrange(self.layer.next.size):
                    self.layer.deltas[i] = self.delta_i(i)
                

    cpdef backwards_pass(self, np.ndarray input_times, np.ndarray desired_times):
        cdef int j, i, k, layer_idx, idx
        cdef int prev_size, next_size
        cdef np.npy_intp *strides
        cdef double *old, *weights
        cdef np.ndarray w
        cdef object layer
        self.forward_pass(input_times, desired_times)
        if self.fail:
            return False
        
        self._delta_j()
        self._delta_i()
        ## Go through every layer backwards
        ## doing the following steps:
        #for layer_idx in xrange(self.layer_length-1, -1, -1):
        
        for layer_idx from self.layer_length-1 >= layer_idx > -1:
            ##  1) figure out what layer we are on
            idx = layer_idx
            layer = self.layers[idx]
            self.layer = layer
            self.last = self.last_layer()

            ##  2) calculate the delta j or delta i depending on the 
            ##     previous step, if this is the last layer we use
            ##     equation 12, if this is input -> hidden, or hidden to hidden
            ##     then we use equation 17

            ##  3) then we go through all of the next neuron set(j) 
            ##     get the time(actual_time), then go through all the 
            ##     neurons in the previous set(i)
            ##     and get the previously calculated delta
            prev = self.layer.prev
            next = self.layer.next

            prev_size = prev.size
            next_size = next.size
            
            weights = <double *>(<np.ndarray>layer.weights).data
            strides = np.PyArray_STRIDES(layer.weights)
            
            for j in xrange(next_size):
                actual_time = next.time[j]
                delta = layer.deltas[j]
                for i in xrange(prev_size):
                    ## 4) from there we go through all the synapses(k)
                    spike_time = prev.time[i]
                    for k in xrange(SYNAPSES):
                        ## 5) we then get the spike time of the last layer(spike_time)
                        ##    get the last weight(old_weight) and if we are on the last
                        ##    layer
                        delay = k + 1
                        old_weight = (<double *>np.PyArray_GETPTR3(layer.weights, i, j, k))[0]
                        delta_weight = self.descent_propagate(i, j, k, actual_time, spike_time, delay, delta)                        
                        new_weight = old_weight + delta_weight
                        IF NEG_WEIGHTS:
                            self.layer.weights[i, j, k] = new_weight
                        ELSE:
                            if new_weight >= 0.0:
                                layer.weights[i, j, k] = new_weight
                            else:
                                layer.weights[i, j, k] = 0.0
                                
        self.layer = layer
        return self.error()
    

    cdef void forward_pass(self, np.ndarray input_times, np.ndarray desired_times):
        ## The first layer will be the furthest most left
        ## and the last layer will be the furthest most right
        ## 0 and -1 respectively
        self.input_layer.prev.time           = input_times
        self.output_layer.next.desired_time  = desired_times
        cdef int h, i, k, z, delay
        cdef int layer_idx
        cdef double time  = 0
        cdef double total = 0
        cdef double ot = 0
        cdef double spike_time, t2, weight, q
        cdef int next_size
        cdef int prev_size
        cdef double *prev_time, *next_time, *weights
        cdef np.ndarray prev_array, next_array, layer_weights
        
        for layer_idx in xrange(self.layer_length):
            self.layer = self.layers[layer_idx]
            self.layer_idx = layer_idx
            #self.last = self.last_layer()
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
                            ot = 0
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

        #return self.error()
                        
    cpdef error(self):
        last_layer = self.layers[-1]
        total = 0.0
        for j in range(last_layer.next.size):
            total += pow((last_layer.next.time[j] - last_layer.next.desired_time[j]), 2.0, None)
            
        return (total/2.0)



if __name__ == "__main__":
    import doctest
    doctest.testmod()
