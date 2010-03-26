# cython: profile=False
# encoding: utf-8
# filename: spikeProp.pyx

import  numpy as np
cimport numpy as np
cimport cython
from python_ref cimport Py_XINCREF, Py_XDECREF, PyObject

DEF DECAY    = 7
DEF SYNAPSES = 16

F_DTYPE = np.float64
ctypedef np.float64_t F_DTYPE_t

cdef extern from "math.h" nogil:
    double c_exp "exp" (double)
    
cdef extern from "spike_prop.h":
    double e(double)
    double srfd(double)
    double link_out(PyObject *weights, double spikeT, double timeT)
    
cdef extern from "stdlib.h" nogil:
    int    c_rand  "rand" ()
    int    c_srand "srand" (int)
    double c_fmod  "fmod" (double, double)

"""
@cython.profile(False)
@cython.boundscheck(False)
cpdef inline double e(double time):
    cdef double t1, t2, t3, t4
    cdef int decay = DECAY
    t4 = 0
    
    if time > 0:
        t1 = (time/decay)
        t2 = c_exp(1-t1)
        t3 = (time*t2)
        t4 = t3/decay
        
    return t4
"""

@cython.profile(False)
@cython.boundscheck(False)
cdef double y(double time, double spike, int delay):
    return e(time-spike-delay)
"""
@cython.boundscheck(False)
cdef double link_out(np.ndarray weights, double spike, double time):
    cdef double weight
    cdef int delay
    cdef int k
    
    sum = 0.0
    r = np.linspace(1, SYNAPSES, num=SYNAPSES)
    condition = time>=(spike+r)
    extract   = np.extract(condition, r)
    if extract.any():
        t = (time-spike-delay)
        b = weights * (t*np.exp(1-t/DECAY))/DECAY
        sum = np.sum(b)

    
    output = 0.0
    if time >= spike:
        for k in range(SYNAPSES):
            weight = weights[k]
            delay = k+1
            if time >= (spike + delay):
                output += (weight * y(time,spike,delay))
            
    #print sum - output
    return output


cpdef double srfd(double time):
    cdef double asrfd = 0
    if time <= 0:
        return asrfd
    else:
        return e(time) * ((1.0/time) - (1.0/DECAY))
"""

cdef class spikeprop:    
    ## without:
    ## python test.py  140.49s user 2.65s system 75% cpu 3:08.61 total
    ## with:
    ## python test.py  130.38s user 2.15s system 75% cpu 2:55.88 total
    
    cdef int synapses
    cdef int inputs
    cdef int hiddens
    cdef int outputs
    cdef int threshold
    cdef int ipsp
    cdef int decay
    cdef int seed

    cdef bool fail
    cdef bool allow_negative_weights
    cdef double learning_rate
    cdef double time_step

    cdef np.ndarray delta_J
    cdef np.ndarray delta_I
    cdef np.ndarray input_time
    cdef np.ndarray hidden_time
    cdef np.ndarray output_time
    cdef np.ndarray desired_time

    cdef np.ndarray output_weights
    cdef np.ndarray hidden_weights
    
    def __init__(self, int inputs, int hiddens, int outputs, int synapses,
                 double learning_rate=0.01, int threshold=50):
        ## Verified:
        ##  Weight Initialisation
        ##  no_adapt
        ##  y
        ##  e
        ##  link_out
        
        self.synapses = synapses
        self.inputs  = inputs
        self.hiddens = hiddens
        self.outputs = outputs
        
        self.fail = False

        self.learning_rate = learning_rate
        self.threshold = threshold
        self.ipsp = 1
        self.time_step = 0.01

        self.allow_negative_weights = False
        
        self.decay = 7
        self.seed  = 3
        
        ## Delta vectors
        self.delta_J = np.ndarray(self.outputs)
        self.delta_I = np.ndarray(self.hiddens)

        ## Time vectors
        self.input_time   = np.zeros(self.inputs)
        self.hidden_time  = np.zeros(self.hiddens)
        self.output_time  = np.zeros(self.outputs)
        self.desired_time = np.zeros(self.outputs)

        ## Weight Matrices
        self.hidden_weights = np.random.rand(self.hiddens, self.inputs, self.synapses).astype(np.float64)*10.0
        self.output_weights = np.random.rand(self.outputs, self.hiddens, self.synapses).astype(np.float64)*10.0
        
    def _fail(self):
        return self.fail
    failed = property(_fail)

    def print_times(self):
        print "Input times,",
        for h in xrange(self.inputs):
            print "input neuron %d: %fms " % (h,self.input_time[h])
        for i in xrange(self.hiddens):
            print "hidden neuron %d: %fms " % (i, self.hidden_time[i])
        for j in xrange(self.outputs):
            print "outputs neuron %d: %fms " % (j, self.output_time[j])
        for j in xrange(self.outputs):
            print "desired neuron %d: %fms " % (j, self.desired_time[j])

    def init_1(self):
        for i in range(self.hiddens):
            for h in range(self.inputs):
                for k in range(self.synapses):
                    r = c_rand() % 10.0
                    self.hidden_weights[i,h,k] = r+1.0

        for i in range(self.outputs):
            for h in range(self.hiddens):
                for k in range(self.synapses):
                    r = c_rand() % 10.0
                    self.output_weights[i,h,k] = r+1.0
                    
    def init_2(self):
        c_srand(self.seed)
        for i in range(self.hiddens):
            for h in range(self.inputs):
                r = c_rand() % 10.0
                for k in range(self.synapses):
                    self.hidden_weights[i,h,k] = r+1.0

        for i in range(self.outputs):
            for h in range(self.hiddens):
                r = c_rand() % 10.0
                for k in range(self.synapses):
                    self.output_weights[i,h,k] = r+1.0
                    
        print self.output_weights
        
    def init_3(self):
        c_srand(self.seed)
        for i in range(self.hiddens):
            for h in range(self.inputs):
                for k in range(self.synapses):
                    rand = c_rand()
                    r = (rand % 10)
                    self.hidden_weights[i,h,k] = 1+(r+1.0)/(14.0*5)

        for j in range(self.outputs):
            for i in range(self.hiddens):
                for k in range(self.synapses):
                    r = (c_rand() % 10)
                    self.output_weights[j,i,k] = 1+(r+1.0)/(14.0*5)
                                   
        print self.hidden_weights

    def loop_all_template(self, h, i, callback):
        cdef Py_ssize_t j,k
        for j in range(i):
            for k in range(h):
                callback()

    #@cython.boundscheck(False)
    cpdef object no_adapt(self, np.ndarray in_times, np.ndarray desired_times):
        self.input_time   = in_times
        self.desired_time = desired_times
        cdef double out = 0.0
        cdef double t   = 0.0
        cdef double spike_time = 0.0
        ## for each neuron find spike time
        ## for each hidden neuron
        for i from 0 <= i < self.hiddens:
            #print "hidden:",
            #print i
            t   = 0.0
            out = 0.0
            while (out < self.threshold and t < 50.0):
                out = 0.0
                ## for each connection to input
                for h from 0 <= h < self.inputs:
                    spike_time = self.input_time[h]
                    if t >= spike_time:
                        out += link_out(<PyObject *>self.hidden_weights[i,h], spike_time, t)
                    
                self.hidden_time[i] = t
                t = t + self.time_step

            if t >= 50.0:
                self.fail = True
                break

        for j from 0 <= j < self.outputs:
            #print "outputs:",
            #print j
            out = 0.0
            t = 0.0
            while (out < self.threshold and t < 50.0):
                out = 0.0
                for i from 0 <= i < self.hiddens:
                    spike_time = self.hidden_time[i]
                    if t >= spike_time:
                        ot = link_out(<PyObject *>self.output_weights[j,i], spike_time, t)
                        if (i >= self.hiddens-self.ipsp):
                            out=out-ot
                        else:
                            out=out+ot
                
                ## End: while
                self.output_time[j] = t
                t += self.time_step

            if t >= 50.0:
                self.fail = True
                break

        return self.error()

    cpdef adapt(self, np.ndarray in_times,  np.ndarray desired_times):
        self.no_adapt(in_times, desired_times)
        if self.fail:
            return False

        #cdef Py_ssize_t h,i,j,k

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
                    if i >= self.hiddens-self.ipsp:
                        change_weight = -self.change(actual_time_j, spike_time, delay, delta)
                    else:
                        change_weight = self.change(actual_time_j, spike_time, delay, delta)
                    
                    new_weight = old_weight + change_weight
                    if self.allow_negative_weights:
                        self.output_weights[j,i,k] = new_weight
                    else:
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
                    if i >= self.hiddens-self.ipsp:
                        change_weight = -self.change(actual_time_i, spike_time, delay, delta)
                    else:
                        change_weight = self.change(actual_time_i, spike_time, delay, delta)

                    new_weight = old_weight + change_weight
                    if self.allow_negative_weights:
                        self.hidden_weights[i,h,k] = new_weight
                    else:
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
            if i >= (self.hiddens - self.ipsp):
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
            if i >= (self.hiddens-self.ipsp):
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

        if i >= (self.hiddens-self.ipsp):
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
            for k in range(self.synapses):
                weight = weights[k]
                delay  = k + 1
                ## will fire when current time 
                ## (timeT) >= time of spike + delay otherwise zero
                if time >= (spike_time + delay):
                    output += (weight * srfd((time - delay - spike_time)))
                ## else no charge

        ## else none will fire
        return output
    """
    def srfd(self, time):
        ## 100%
        if time >= 0.0:
            return e(time) *((1.0/time) - (1.0/self.decay))
        else:
            return 0.0
    """
    def error(self):
        cdef double total = 0.0
        for j in range(self.outputs):
            total += (self.output_time[j]-self.desired_time[j]) ** 2

        return (total/2)



def xor(which):
    if which == 0:
        input = np.array([0.0, 0.0, 0.0])
        desired = np.array([16.0])
    elif which == 1:
        input = np.array([0.0, 0.0, 6.0])
        desired = np.array([10.0])
    elif which == 2:
        input = np.array([0.0, 6.0, 0.0])
        desired = np.array([10.0])
    elif which == 3:
        input = np.array([0.0, 6.0, 6.0])
        desired = np.array([16.0])
    else:
        input = np.array([0.0, 6.0, 6.0])
        desired = np.array([16.0])

    return input, desired



