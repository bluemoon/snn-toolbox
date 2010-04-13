# cython: profile=False
# cython: boundscheck=False
# cython: wraparound=False
# encoding: utf-8
# filename: spikeprop_ng.pyx

## import multiprocessing as mp
## CPU_CORES = mp.cpu_cores() 
import  numpy as np
cimport numpy as np
cimport cython as cy
cimport python as py



## Initialise the C-API for numpy
np.import_array()

DEF DECAY       = 7
DEF SYNAPSES    = 16
DEF IPSP        = 1
DEF MAX_TIME    = 50
DEF TIME_STEP   = 0.01
DEF NEG_WEIGHTS = False
DEF MP          = True

cdef extern from "math.h" nogil:
    double c_exp "exp" (double)
    double c_modf "modf" (double, double*)
    
cdef extern from "spike_prop.h" nogil:
    double e(double)
    
cdef extern from "stdlib.h" nogil:
    int    c_rand  "rand" ()
    int    c_srand "srand" (int)
    double c_fmod  "fmod" (double, double)

IF 0:
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

@cy.profile(False)
@cy.cdivision(True)
cdef double srfd(double time) nogil:
    cdef double asrfd = 0
    if time <= 0:
        return asrfd
    else:
        return e(time) * ((1.0/time) - (1.0/DECAY))

@cy.profile(False)
@cy.boundscheck(False)
cdef double y(double time, double spike, int delay) nogil:
    return e(time-spike-delay)

cdef class spikeprop_faster:
    cdef int seed, inputs, hiddens, outputs,
    cdef int threshold
    cdef bint fail
    cdef double learning_rate
    cdef np.ndarray hidden_time, output_time, desired_time, input_time
    cdef np.ndarray hidden_weights, output_weights
    cdef np.ndarray delta_J, delta_I

    def __init__(self, object inputs, object hiddens, object outputs):
        self.inputs  = inputs
        self.hiddens = hiddens
        self.outputs = outputs

        self.threshold = 50
        ## Messing around with optimization below
        ## i_time = np.PyArray_ZEROS(self.input_time,  inputs,  np.NPY_DOUBLE, 0)
        ## print i_time
        ## np.PyArray_ZEROS(self.hidden_time,  hiddens, np.NPY_DOUBLE, 0)
        ## np.PyArray_ZEROS(self.output_time,  outputs, np.NPY_DOUBLE, 0)
        ## np.PyArray_ZEROS(self.desired_time, outputs, np.NPY_DOUBLE, 0)
        ## self.input_time   = np.PyArray_Zeros(1, [inputs], np.float64_t)#np.zeros(inputs)

        ## Time Vectors
        ################
        self.input_time   = np.zeros(inputs)
        self.hidden_time  = np.zeros(hiddens)
        self.output_time  = np.zeros(outputs)
        self.desired_time = np.zeros(outputs)

        ## Failure state switch
        ## This acts as a switch that we check to see
        ## if it has failed yet
        self.fail = False
        
        ## Weight initialisation
        #########################
        self.hidden_weights = np.random.rand(self.hiddens, self.inputs, SYNAPSES).astype(np.float64)*10.0
        self.output_weights = np.random.rand(self.outputs, self.hiddens, SYNAPSES).astype(np.float64)*10.0

        ## Delta vectors
        #################
        self.delta_J = np.ndarray(self.outputs)
        self.delta_I = np.ndarray(self.hiddens)

        ## Learning rate
        #################
        self.learning_rate = 1.0
        
    cpdef bint _fail(self):
        return self.fail
    failed = property(_fail)
    
    @cy.boundscheck(False)
    cdef double link_out(self, np.ndarray weights, double spike, double time):
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
    
    cdef sub1(self, k):
        pass

    cpdef initialise_weights(self):
        c_srand(self.seed)
        for i from 0 <= i < self.hiddens:
            for h from 0 <= h < self.inputs:
                r = c_fmod(c_rand(), 10.0)
                for k from 0 <= k < SYNAPSES:
                    print i,h,k
                    self.hidden_weights[i,h,k] = r+1.0

        for i in range(self.outputs):
            for h in range(self.hiddens):
                r = c_fmod(c_rand(), 10.0)
                for k in range(SYNAPSES):
                    self.output_weights[i,h,k] = r+1.0


        

    cpdef forward_pass(self, np.ndarray input, np.ndarray desired):
        return self._forward_pass(input, desired)

    cdef _forward_pass(self, np.ndarray in_times, np.ndarray desired_times):
        self.input_time   = in_times
        self.desired_time = desired_times
        cdef double out = 0.0, t = 0.0, spike_time = 0.0
        
        cdef double *in_time = <double *>self.input_time.data
        cdef double *hidden_time = <double *>self.hidden_time.data
        cdef double *output_time = <double *>self.output_time.data
        
        #cdef double *hw = <double *>self.hidden_weights.data
        cdef int i=0, h=0, j=0
        ## for each neuron find spike time
        ## for each hidden neuron
        for i from 0 <= i < self.hiddens:
            ## reset time and output total
            t   = 0
            out = 0
            while (out < self.threshold and t < MAX_TIME):
                out = 0
                ## for each connection to input
                for h from 0 <= h < self.inputs:
                    spike_time = in_time[h]
                    if t >= spike_time:
                        out += self.link_out(self.hidden_weights[py.PyInt_FromLong(i), py.PyInt_FromLong(h)], spike_time, t)
                    
                hidden_time[i] = t
                t += TIME_STEP

            if t >= 50.0:
                self.fail = True
                break

        for j from 0 <= j < self.outputs:
            out = 0.0
            t = 0.0
            while (out < self.threshold and t < MAX_TIME):
                out = 0.0
                for i from 0 <= i < self.hiddens:
                    spike_time = hidden_time[i]
                    if t >= spike_time:
                        ot = self.link_out(self.output_weights[j,i], spike_time, t)
                        if (i >= self.hiddens-IPSP):
                            out=out-ot
                        else:
                            out=out+ot
                
                ## End: while
                output_time[j] = t
                t += TIME_STEP

            if t >= 50.0:
                self.fail = True
                break

        return self.error()
    
    cpdef adapt(self, np.ndarray in_times,  np.ndarray desired_times):
        self._forward_pass(in_times, desired_times)
        if self.fail:
            return False

        ## for each output neuron
        for j from 0 <= j < self.outputs:
            self.delta_J[j] = self._e12(j)
        ## for each hidden neuron
        for i from 0 <= i < self.hiddens:
            self.delta_I[i] = self._e17(i)
        
        ## For each output neuron
        for j from 0 <= j < self.outputs:
            actual_time_j = self.output_time[j]
            delta = self.delta_J[j]
            ## For each connection to hidden
            for i from 0 <= i < self.hiddens:
                ## Spike time of the hidden neuron
                spike_time = self.hidden_time[i]
                for k from 0 <= k < SYNAPSES:
                    delay = k+1
                    old_weight = self.output_weights[j,i,k]
                    if i >= self.hiddens-IPSP:
                        change_weight = -self.change(actual_time_j, spike_time, delay, delta)
                    else:
                        change_weight = self.change(actual_time_j, spike_time, delay, delta)
                    
                    new_weight = old_weight + change_weight
                    #if self.allow_negative_weights:
                    IF NEG_WEIGHTS:
                        self.output_weights[j,i,k] = new_weight
                    ELSE:
                        if new_weight >= 0.0:
                            self.output_weights[j,i,k] = new_weight
                        else:
                            self.output_weights[j,i,k] = 0.0
        
        for i from 0 <= i < self.hiddens:
            actual_time_i = self.hidden_time[i]
            delta = self.delta_I[i]
            for h from 0 <= h < self.inputs:
                spike_time = self.input_time[h]
                for k from 0 <= k < SYNAPSES:
                    delay=k+1
                    old_weight=self.hidden_weights[i,h,k]
                    if i >= self.hiddens-IPSP:
                        change_weight = -self.change(actual_time_i, spike_time, delay, delta)
                    else:
                        change_weight = self.change(actual_time_i, spike_time, delay, delta)

                    new_weight = old_weight + change_weight
                    IF NEG_WEIGHTS:
                        self.hidden_weights[i,h,k] = new_weight
                    ELSE:
                        if new_weight >= 0.0:
                            self.hidden_weights[i,h,k] = new_weight#new_weight
                        else:
                            self.hidden_weights[i,h,k] = 0.0

        return self.error()
    
    cdef _e12(self, j):
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
 
    cdef _e17top(self, i, delta_j):
        #cdef double ot = 0.0
        #cdef double actual = 0.0
        ot = 0.0
        actual = 0.0
        spike_time = self.hidden_time[i]
        for j in range(self.outputs):
            actual_time_j = self.output_time[j]
            dj = delta_j[j]
            if i >= (self.hiddens-IPSP):
                ot = -self.link_out_d(self.output_weights[j,i], spike_time, actual_time_j)
            else:
                ot = self.link_out_d(self.output_weights[j,i], spike_time, actual_time_j)
            actual = actual + (dj*ot)

        return actual
    
    cdef _e17bottom(self, i):
        ## 100%
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

    cdef _e17(self, i):
        ## 100%
        actual = self._e17top(i, self.delta_J)/self._e17bottom(i)
        return actual
            

    cdef change(self, actual_time, spike_time, delay, delta):
        return (-self.learning_rate * y(actual_time, spike_time, delay) * delta)

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

        ## else none will fire
        return output

    cpdef error(self):
        cdef double total = 0.0
        for j in range(self.outputs):
            total += (self.output_time[j]-self.desired_time[j]) ** 2

        return (total/2)
