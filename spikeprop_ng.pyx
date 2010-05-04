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
DEF TIME_STEP   = 0.1
DEF NEG_WEIGHTS = False
DEF MP          = True

DEF QUICKPROP   = False
DEF RPROP       = False

class Inputter:
    def __init__(self, data, type):
        self.data = data
        self.type = type


cdef int sign(num):
    if num < 0:
        return -1
    elif num > 0:
        return 1
    else:
        return 0
    
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



cdef class spikeprop_faster:
    cdef int seed, inputs, hiddens, outputs
    cdef public int threshold
    cdef bint fail
    cdef double learning_rate
    cdef np.ndarray hidden_time, output_time, desired_time, input_time
    cdef np.ndarray hidden_weights, output_weights
    cdef np.ndarray delta_J, delta_I
    cdef public np.ndarray h_derive, o_derive, h_delta, o_delta
    #cdef np.ndarray h_learn_rates, o_learn_rates
    
    cdef int with_ipsp

    def __init__(self, object inputs, object hiddens, object outputs):
        self.inputs  = inputs
        self.hiddens = hiddens
        self.outputs = outputs

        self.threshold = 50
        
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
        
        self.o_derive = np.zeros((self.outputs, self.hiddens, SYNAPSES))
        self.h_derive = np.zeros((self.hiddens, self.inputs, SYNAPSES))
        self.o_delta = np.ones((self.outputs, self.hiddens, SYNAPSES))
        self.h_delta = np.ones((self.hiddens, self.inputs, SYNAPSES))
        
        ## Learning rate
        #################
        self.learning_rate = 1.0
        
    cpdef save_weights(self, fname):
        FILE = open(fname, 'w')
        cp.dump([self.hidden_weights, self.output_weights], FILE)
        FILE.close()
        
    cpdef load_weights(self, fname):
        if os.path.exists(fname):
            FILE = open(fname, 'r')
            weights = cp.load(FILE)
            FILE.close()
            
            self.hidden_weights = weights[0]
            self.output_weights = weights[1]
        else:
            print "!!!! FAILED !!!!"
            
    cpdef bint _fail(self):
        return self.fail
    failed = property(_fail)
    
    #cdef weight_minmax(self, inputs):
    #    t_min = 0.1
    #    t_max = 10
    #    return self._weight_minmax(inputs, t_min), self._weight_minmax(inputs, t_max)
   # 
   # cdef _weight_minmax(self, inputs, time):
    #    return (DECAY*self.threshold)/(SYNAPSES*inputs*time)*c_exp((time/DECAY)-1.0)
        
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
    

    cpdef initialise_weights(self):
        cdef int Type = 0
        cdef int i,h,k
        if Type == 0:
            c_srand(self.seed)
            for i from 0 <= i < self.hiddens:
                for h from 0 <= h < self.inputs:
                    r = c_fmod(c_rand(), 10.0)
                    for k from 0 <= k < SYNAPSES:
                        self.hidden_weights[i,h,k] = (r+1.0)

            for i in range(self.outputs):
                for h in range(self.hiddens):
                    r = c_fmod(c_rand(), 10.0)
                    for k in range(SYNAPSES):
                        self.output_weights[i,h,k] = (r+1.0)
                        
        if Type == 1:
            mu, sigma = 1.11, 0.35
            self.hidden_weights = np.random.normal(mu, sigma, size=(self.hiddens, self.inputs, SYNAPSES))
            self.output_weights = np.random.normal(mu, sigma, size=(self.outputs, self.hiddens, SYNAPSES))


        


    cpdef forward_pass(self, np.ndarray input, np.ndarray desired):
        ## for overhead of python passing
        return self._forward_pass(input, desired)

    cdef _forward_pass(self, np.ndarray in_times, np.ndarray desired_times):
        ## the c-specific one incurs less overhead than the python
        ## specific one
        self.input_time   = in_times
        self.desired_time = desired_times
        cdef double out = 0.0, time = 0.0, spike_time = 0.0
        
        cdef double *in_time = <double *>self.input_time.data
        cdef double *hidden_time = <double *>self.hidden_time.data
        cdef double *output_time = <double *>self.output_time.data
        
        #cdef double *hw = <double *>self.hidden_weights.data
        cdef int i=0, h=0, j=0
        ## for each neuron find spike time
        ## for each hidden neuron
        for i from 0 <= i < self.hiddens:
            ## reset time and output total
            time = 0
            out  = 0
            while (out < self.threshold and time < MAX_TIME):
                out = 0
                ## for each connection to input
                for h from 0 <= h < self.inputs:
                    spike_time = in_time[h]
                    if time >= spike_time:
                        out += self.link_out(self.hidden_weights[i, h], spike_time, time)
                    
                hidden_time[i] = time
                time += TIME_STEP

            if time >= 50.0:
                self.fail = True
                break

        for j from 0 <= j < self.outputs:
            out  = 0.0
            time = 0.0
            while (out < self.threshold and time < MAX_TIME):
                out = 0.0
                for i from 0 <= i < self.hiddens:
                    spike_time = hidden_time[i]
                    if time >= spike_time:
                        ot = self.link_out(self.output_weights[j,i], spike_time, time)
                        ## if we are on the IPSP synapses
                        if (i >= self.hiddens-IPSP):
                            ## if we are depress the synapse
                            out=out-ot
                        else:
                            out=out+ot
                
                ## End: while
                output_time[j] = time
                time += TIME_STEP

            if time >= 50.0:
                self.fail = True
                break

        return self.error()
    
    cpdef clear_slopes(self):
        decay = -0.001
        #IF QUICKPROP:
        #    self.o_derive = decay * self.output_weights
        #    self.h_derive = decay * self.hidden_weights 

        
    
    cpdef adapt(self, np.ndarray in_times,  np.ndarray desired_times):
        self._forward_pass(in_times, desired_times)
        if self.fail:
            return False
        
        cdef int i, j, k
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
                    delay = k + 1
                    old_weight = self.output_weights[j,i,k]
                    ##   ∂E
                    ## -------  = ε.ij^k(t-t.i-d.ij^k)δ.j
                    ## ∂w.ij^k
                    IF QUICKPROP:
                        ## for quickprop the math is
                        ## pde E
                        ##------ of time
                        ## pde w
                        
                        E_double_prime = self.o_derive[j,i,k]
                        E_prime = self.error_weight_derivative(actual_time_j, spike_time, delay, delta)
                        if (sign(E_prime) == sign(E_double_prime)):
                            ## if the signs are the same do a simple gradient
                            ## descent
                            momentum = 0.0
                            change_weight = -self.learning_rate * (E_prime) + (momentum * self.o_delta[j,i,k])
                        else:
                            change_weight = (E_prime /(E_double_prime-E_prime)) * self.o_delta[j,i,k]
                            
                        new_weight = old_weight + change_weight
                        self.output_weights[j,i,k] = new_weight
                        self.o_delta[j,i,k] = change_weight
                        self.o_derive[j,i,k] = E_prime
                        
                    ELIF RPROP:
                        E_double_prime = self.o_derive[j,i,k]
                        E_prime = self.error_weight_derivative(actual_time_j, spike_time, delay, delta)
                        rprop = E_prime * E_double_prime
                        
                        if rprop > 0:
                            change_weight = E_prime * 1.3
                        elif rprop < 0:
                            change_weight = E_prime * 0.5
                        else:
                            change_weight = E_prime
                            
                        if E_prime > 0:
                            change_weight = -change_weight
                        
                        new_weight = old_weight + change_weight
                        self.output_weights[j,i,k] = new_weight
                        self.o_derive[j,i,k] = E_prime
                        
                    ELSE:
                        ## α is momentum
                        if i >= self.hiddens-IPSP:
                            change_weight = -self.change(actual_time_j, spike_time, delay, delta)
                        else:
                            change_weight = self.change(actual_time_j, spike_time, delay, delta)
                            
                        #new_weight = old_weight / (self.o_weight_last[j,i,k]-old_weight)
                        new_weight = old_weight + change_weight 
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
                    delay = k + 1
                    old_weight = self.hidden_weights[i, h, k]
                    IF QUICKPROP:
                        E_double_prime = self.h_derive[i,h,k]
                        E_prime = self.error_weight_derivative(actual_time_i, spike_time, delay, delta)
                        if sign(E_double_prime) == sign(E_prime) or E_double_prime == 0:
                            momentum = 0.0
                            change_weight = -(self.learning_rate * E_prime + (momentum * self.h_delta[i,h,k]))
                        else:
                            change_weight = (E_prime/(E_double_prime-E_prime))  * self.h_delta[i,h,k]
                            
                        new_weight = old_weight + change_weight
                        self.h_delta[i,h,k]  = change_weight
                        self.h_derive[i,h,k] = E_prime
                        self.hidden_weights[i,h,k] = new_weight

                    ELIF RPROP:
                        E_double_prime = self.h_derive[i,h,k]
                        E_prime = self.error_weight_derivative(actual_time_i, spike_time, delay, delta)
                        rprop = E_prime * E_double_prime
                        
                        if rprop > 0:
                            change_weight = E_prime * 1.3
                        elif rprop < 0:
                            change_weight = E_prime * 0.5
                        else:
                            change_weight = E_prime

                        if E_prime > 0:
                            change_weight = -change_weight
                        
                        new_weight = old_weight + change_weight
                        self.hidden_weights[i,h,k] = new_weight#new_weight    
                        self.h_derive[i,h,k] = E_prime
                    ELSE:
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
                            
        #IF QUICKPROP:
        #    self.o_weight_last = self.output_weights
        #    self.h_weight_last = self.hidden_weights
            
        return self.error()
    
    cdef error_weight_derivative(self, actual_time, spike_time, delay, delta):
        return y(actual_time, spike_time, delay) * delta
    
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
            ot = self.link_out_d(self.hidden_weights[i, h], spike_time, actual_time)
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
        total = 0.0
        for j in range(self.outputs):
            total += ((self.output_time[j]-self.desired_time[j]) ** 2.0)
            
        return (total/2.0)
