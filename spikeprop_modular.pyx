# cython: profile=False
# cython: boundscheck=False
# cython: wraparound=False
# cython: infer_types=True
# encoding: utf-8
# filename: spikeprop_ng.pyx

## import multiprocessing as mp
## CPU_CORES = mp.cpu_cores() 
import os

from spikeprop  cimport *

import  numpy  as np
import cPickle as cp
cimport numpy  as np
cimport cython as cy
cimport python as py

## V is a set of spiking neurons
## V ⊆ X
## V ⊆ Y
## X = input
## Y = output

## Initialise the C-API for numpy
np.import_array()

## All of the defines that are substituted in at compile time
DEF DECAY       = 7
DEF SYNAPSES    = 16
DEF IPSP        = 1
DEF MAX_TIME    = 50
DEF TIME_STEP   = 0.01
DEF NEG_WEIGHTS = False
DEF MP          = True

DEF QUICKPROP   = False
DEF RPROP       = False

@cy.cdivision(True)
cdef double srfd(double time) nogil:
    cdef double asrfd = 0
    if time <= 0:
        return asrfd
    else:
        return e(time) * ((1.0/time) - (1.0/DECAY))

cdef double y(double time, double spike, int delay) nogil:
    return e(time-spike-delay)

@cy.boundscheck(False)
cdef double link_out(np.ndarray weights, double spike, double time):
    cdef double *p = <double *>weights.data
    cdef double weight, output = 0.0
    cdef int k, i, delay
    ## if time >= (spike + delay)
    ## delay_max = SYNAPSES
    ## the delay is 1...16
    ## if time >= (spike + {1...16})
    ## so i need to find the minimum delay size and
    ## start the loop from there

    i = int(time-spike)
    for k from 0 <= k < i:
        delay = k+1
        weight = p[k]
        output += (weight * e(time-spike-delay))

    return output

cdef link_out_d(self, np.ndarray weights, double spike_time, double time):
    ## 100%
    cdef double output = 0.0
    cdef int delay
    cdef Py_ssize_t k

    #output = 0.0
    if time >= spike_time:
        for k in range(SYNAPSES):
            weight = weights[k]
            delay  = k + 1
            ## will fire when current time 
            ## (timeT) >= time of spike + delay otherwise zero
            if time >= (spike_time + delay):
                output += (weight * srfd((time - delay - spike_time)))
                ## else no charge


cdef class neurons:
    cdef public np.ndarray time
    cdef public np.ndarray desired
    
    def __init__(self, neurons):
        self.time    = np.ndarray((neurons))
        self.desired = np.ndarray((neurons))
    
class layer:
    def __init__(self, in_neurons, out_neurons):
        self.ins  = in_neurons
        self.outs = out_neurons
        self.weights = np.random.rand(self.ins, self.outs, SYNAPSES) * 10.0
        self.delays  = np.random.rand(self.ins, self.outs)
        self.deltas  = np.random.rand(self.ins, self.outs)
        self.In  = neurons(self.ins)
        self.Out = neurons(self.outs)
        self.learning_rate = 1.0
        

cdef class modular:
    cdef object layers
    cdef object layer
    cdef int threshold
    cdef public bint failed
    cdef np.ndarray desired_time
    
    def __init__(self, layers):
        self.layers    = layers
        self.threshold = 50
        self.failed = False
        self.layer = None
    
    cpdef backwards_pass(self, np.ndarray input, np.ndarray desired):
        self.desired_time = desired
        self._forward_pass(input, desired)
        for layer_idx from 0 <= layer_idx < len(self.layers):
            self.layer = self.layers[layer_idx]
            for i from 0 <= i < self.layer.outs:
                if layer_idx < len(self.layers):
                    self.layer.deltas[i] = self.equation_17(i)
                elif layer_idx == len(self.layers):
                    self.layer.deltas[i] = self.equation_12(i)
                    
                
            for j from 0 <= j < self.layer.outs:
                actual_time = self.layer.Out.time[j]
                for i from 0 <= i < self.layer.ins:
                    for k from 0 <= k < SYNAPSES:
                        delta = self.layer.deltas[j]
                        delay = k+1

                        spike_time = self.layer.In.time[i]
                        old_weight = self.layer.weights[i,j,k]
                        
                        if i >= self.layer.outs-IPSP:
                            change_weight = -self.change(actual_time, spike_time, delay, delta)
                        else:
                            change_weight = self.change(actual_time, spike_time, delay, delta)

                        new_weight = old_weight + change_weight
                            
                        IF NEG_WEIGHTS:
                            self.layer.weights[i,j,k] = new_weight
                        ELSE:
                            if new_weight >= 0.0:
                                self.layer.weights[i,j,k] = new_weight#new_weight
                            else:
                                self.layer.weights[i,j,k] = 0.0
        return self.error()
                    
    cpdef forward_pass(self, np.ndarray input, np.ndarray desired):
        ## for overhead of python passing
        return self._forward_pass(input, desired)

    cdef _forward_pass(self, np.ndarray in_times, np.ndarray desired_times):
        ## the c-specific one incurs less overhead than the python
        ## specific one
        self.layers[0].In.time      = in_times
        self.layers[-1].Out.desired = desired_times
        
        total = 0
        for layer_idx from 0 <= layer_idx < len(self.layers):
            self.layer = self.layers[layer_idx]
            if layer_idx == 0:
                for i from 0 <= i < self.layer.outs:
                    total = 0
                    time  = 0
                    while (total < self.threshold and time < MAX_TIME):
                        for h from 0 <= h < self.layer.ins:
                            spike_time = in_times[h]
                            if time >= spike_time:
                                total += link_out(self.layer.weights[h,i], spike_time, time)
                                
                        self.layer.Out.time[i] = time
                        time += TIME_STEP
                    if time >= 50.0:
                        self.failed = True
                        
            if layer_idx > 0:
                for i from 0 <= i < self.layer.outs:
                    total = 0
                    time  = 0
                    while (total < self.threshold and time < MAX_TIME):
                        for h from 0 <= h < self.layer.ins:
                            spike_time = in_times[i]
                            if time >= spike_time:
                                ot = link_out(self.layer.weights[h,i], spike_time, time)
                                if (i >= self.layer.outs-IPSP):
                                    total=total-ot
                                else:
                                    total=total+ot
                                    
                        self.layer.Out.time[i] = time
                        time += TIME_STEP
                        
                    if time >= 50.0:
                        self.failed = True
                        

        
    
    cdef equation_12(self, j):
        return (self.desired_time[j]-self.output_time[j])/(self._e12bottom(j))

    cdef _e12bottom(self, j):
        ot = 0.0
        for i in range(self.hiddens):
            if i >= (self.hiddens - IPSP):
                ot = ot - self.link_out_d(self.output_weights[j,i], \
                self.hidden_time[i], self.output_time[j])
            else:
                ot = ot + self.link_out_d(self.output_weights[j,i], \
                self.hidden_time[i], self.output_time[j])

        return ot
 
    cdef equation_17_top(self, i, delta_j):
        ot = 0.0
        actual = 0.0
        
        spike_time = self.layer.Out.time[i]
        for j in range(self.layers[-1].outs):
            actual_time_j = self.layers[-1].Out.time[j]
            dj = delta_j[j]
            if i >= (self.layer.outs-IPSP):
                ot = -self.link_out_d(self.layer.weights[j,i], spike_time, actual_time_j)
            else:
                ot = self.link_out_d(self.layer.weights[j,i], spike_time, actual_time_j)
            actual = actual + (dj*ot)

        return actual
    
    cdef equation_17_bottom(self, i):
        cdef double actual = 0.0
        cdef double ot, actual_time = 0.0
        actual_time = self.hidden_time[i]

        for h in range(self.inputs):
            spike_time = self.input_time[h]
            ot = self.link_out_d(self.hidden_weights[i,h], spike_time, actual_time)
            actual = actual + ot
        
        if i >= (self.hiddens-IPSP):
            return -actual
        else:
            return actual

    cdef equation_17(self, i):
        actual = self.equation_17_top(i, self.layer.deltas[i])/self.equation_17_bottom(i)
        return actual
            

    cdef change(self, actual_time, spike_time, delay, delta):
        return (-self.layer.learning_rate * y(actual_time, spike_time, delay) * delta)

    cdef link_out_d(self, np.ndarray weights, double spike_time, double time):
        cdef double output = 0.0
        cdef int delay
        cdef Py_ssize_t k

        #output = 0.0
        if time >= spike_time:
            for k in range(SYNAPSES):
                weight = weights[k]
                delay  = k + 1
                ## will fire when current time 
                ## (timeT) >= time of spike + delay otherwise zero
                if time >= (spike_time + delay):
                    output += (weight * srfd((time - delay - spike_time)))
                ## else no charge

        ## else none will fire
        return output

    cdef error(self):
        last_layer = self.layers[-1]
        total = 0.0
        for j in range(last_layer.outs):
            total += ((last_layer.Out.time[j]-last_layer.Out.desired[j]) ** 2.0)
            
        return (total/2.0)
