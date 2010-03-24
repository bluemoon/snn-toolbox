# cython: profile=False
# encoding: utf-8
# filename: spikeProp.pyx

import  numpy as np
cimport numpy as np
cimport cython

F_DTYPE = np.float64
ctypedef np.float64_t F_DTYPE_t

cdef extern from "math.h" nogil:
    double c_exp "exp" (double)
    
cdef extern from "stdlib.h" nogil:
    int    c_rand "rand" ()
    int    c_srand "srand" (int)
    double c_fmod "fmod" (double, double)

@cython.profile(False)
cpdef inline double e(double time, int decay):
    cdef double e = 0
    if time > 0:
        e = ((time/decay)*c_exp(1.0-(time/decay)))
    return e

@cython.profile(False)
cpdef inline double y(double time, double spike_time, int delay, int decay):
    return e((time-spike_time-delay), decay)

@cython.profile(False)
def link_out(np.ndarray[F_DTYPE_t] weights, double time, double spike, int synapses, int decay):
    assert weights.dtype == F_DTYPE
    
    cdef double output  = 0.0
    cdef double sum
    cdef double y  
    cdef double weight
    cdef int delay
    cdef int k 

    if time >= spike:
        for k in range(synapses):
            weight = weights[k]
            delay = k+1
            sum = (spike + delay)
            if time >= sum:
                y = e((time-spike-delay), decay)
                sum = (weight * y)
                output += sum
                
    return output

class spikeprop:    
    ## without:
    ## python test.py  140.49s user 2.65s system 75% cpu 3:08.61 total
    ## with:
    ## python test.py  130.38s user 2.15s system 75% cpu 2:55.88 total
    """
    cdef int synapses
    cdef int inputs
    cdef int hiddens
    cdef int outputs
    cdef int threshold
    cdef int ipsp
    cdef int decay

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
    """
    def __init__(self, int inputs, int hiddens, int outputs, int synapses,
                 double learning_rate=0.1, int threshold=50):
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
        
        self.decay =  7

        ## Delta vectors
        self.delta_J = np.ndarray(self.outputs)
        self.delta_I = np.ndarray(self.hiddens)

        ## Time vectors
        self.input_time   = np.zeros(self.inputs)
        self.hidden_time  = np.zeros(self.hiddens)
        self.output_time  = np.zeros(self.outputs)
        self.desired_time = np.zeros(self.outputs)

        ## Weight Matrices
        self.hidden_weights = np.random.rand(self.hiddens, self.inputs, self.synapses)*10
        self.output_weights = np.random.rand(self.outputs, self.hiddens, self.synapses)*10
        #self.hidden_weights = np.ndarray(shape=(self.hiddens, self.inputs, self.synapses))
        #self.output_weights = np.ndarray(shape=(self.outputs, self.hiddens, self.synapses))
        
        
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

    def init_2(self):
        for i in range(self.hiddens):
            for h in range(self.inputs):
                r = c_fmod(c_rand(), 10)
                for k in range(self.synapses):
                    self.hidden_weights[i,h,k] = np.float64(r+1)

        for i in range(self.outputs):
            for h in range(self.hiddens):
                r = c_fmod(c_rand(), 10)
                print r
                for k in range(self.synapses):
                    self.output_weights[i,h,k] = np.float64(r+1)
                    
    def init_3(self):
        c_srand(4)
        for i in range(self.hiddens):
            for h in range(self.inputs):
                rand = c_rand()
                r = (rand % 10)
                for k in range(self.synapses):
                    self.hidden_weights[i,h,k] = (r+1.0)/(14.0*5)

        for j in range(self.outputs):
            for i in range(self.hiddens):
                r = (c_rand() % 10)
                for k in range(self.synapses):
                    self.output_weights[j,i,k] = (r+1.0)/(14.0*5)
                                   
        print self.hidden_weights
        
    def no_adapt(self, in_times, desired_times):
        cdef Py_ssize_t h,i,j,k
        self.input_time   = in_times
        self.desired_time = desired_times
        cdef float out = 0.0
        cdef float t = 0.0

        ## for each neuron find spike time
        ## for each hidden neuron
        for i in range(self.hiddens):
            t = 0.0
            out = 0.0
            while (out < self.threshold and t < 50.0):
                out = 0.0
                ## for each connection to input
                for h in range(self.inputs):
                    spike_time = self.input_time[h]
                    if t >= spike_time:
                        print self.hidden_weights[i,h].ndim
                        out += link_out(self.hidden_weights[i,h], t, spike_time, self.synapses, self.decay)
                self.hidden_time[i] = t
                t += self.time_step

            if t >= 50.0:
                self.fail = True
                break

        for j in range(self.outputs):
            out = 0.0
            t = 0.0
            while (out < self.threshold and t < 50.0):
                out = 0.0
                for i in range(self.hiddens):
                    spike_time = self.hidden_time[i]
                    if t >= spike_time:
                        ot = link_out(self.output_weights[j,i], t, spike_time, self.synapses, self.decay)
                        if i >= (self.hiddens-self.ipsp):
                            out=out-ot
                        else:
                            out=out+ot


                self.output_time[j] = t
                t += self.time_step             

            if t >= 50.0:
                self.fail = True
                break

        return self.error()

    def adapt(self, in_times,  desired_times):
        self.no_adapt(in_times, desired_times)
        if self.fail:
            return False

        cdef Py_ssize_t h,i,j,k

        ## for each output neuron
        for j in xrange(self.outputs):
            self.delta_J[j] = self._e12(j)
        ## for each hidden neuron
        for i in xrange(self.hiddens):
            self.delta_I[i] = self._e17(i, self.delta_J)
        
        ## For each output neuron
        for j in range(self.outputs):
            actual_time_j = self.output_time[j]
            delta = self.delta_J[j]
            ## For each connection to hidden
            for i in range(self.hiddens):
                ## Spike time of the hidden neuron
                spike_time = self.hidden_time[i]
                for k in range(self.synapses):
                    delay = k+1
                    old_weight = self.output_weights[j,i,k]
                    if i >= (self.hiddens-self.ipsp):
                        change_weight = -self.change(actual_time_j, spike_time, delay, delta)
                    else:
                        change_weight = self.change(actual_time_j, spike_time, delay, delta)
                    
                    new_weight = old_weight + change_weight
                    if self.allow_negative_weights:
                        self.output_weights[j,i,k] = new_weight
                    else:
                        if new_weight >= 0:
                            self.output_weights[j,i,k] = new_weight
                        else:
                            self.output_weights[j,i,k] = 0
        
        for i in xrange(self.hiddens):
            actual_time_i = self.hidden_time[i]
            delta = self.delta_I[i]
            for h in xrange(self.inputs):
                spike_time = self.input_time[h]
                for k in xrange(self.synapses):
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
                        if new_weight >= 0:
                            self.hidden_weights[i,h,k] = new_weight
                        else:
                            self.hidden_weights[i,h,k] = 0

        return self.error()

    


    

    def _e12(self, j):
        return (self.desired_time[j]-self.output_time[j])/(self._e12bottom(j))

    def _e12bottom(self, j):
        ot = 0.0
        for i in range(self.hiddens):
            if i >= (self.hiddens - self.ipsp):
                ot -=  self.link_out_d(self.output_weights[j,i], \
                self.hidden_time[i], self.output_time[j])
            else:
                ot +=  self.link_out_d(self.output_weights[j,i], \
                self.hidden_time[i], self.output_time[j])

        return ot
 
    def _e17top(self, i, delta_j):
        cdef double ot = 0.0
        cdef double actual = 0.0
        spike_time = self.hidden_time[i]
        for j in range(self.outputs):
            actual_time_j = self.output_time[j]
            dj = delta_j[j]
            if i >= (self.hiddens-self.ipsp):
                ot = -self.link_out_d(self.output_weights[j,i], spike_time, actual_time_j)
            else:
                ot = -self.link_out_d(self.output_weights[j,i], spike_time, actual_time_j)
            actual += (dj*ot)

        return actual
    
    def _e17bottom(self, i):
        ## 100%
        cdef double actual = 0.0
        cdef double ot, actual_time = 0.0
        actual_time = self.hidden_time[i]

        for h in range(self.inputs):
            spike_time = self.input_time[h]
            ot = self.link_out_d(self.hidden_weights[i,h], spike_time, actual_time)
            actual += ot

        if i >= (self.hiddens-self.ipsp):
            return -actual
        else:
            return actual

    def _e17(self, i, delta_j):
        ## 100%
        actual = self._e17top(i, delta_j)/self._e17bottom(i)
        return actual
            

    def change(self, actual_time, spike_time, delay, delta):
        return (-self.learning_rate * y(actual_time, spike_time, delay, self.decay)*delta)

    def link_out_d(self, weights, spike_time, time):
        ## 100%
        cdef double output = 0.0
        cdef int delay
        cdef Py_ssize_t k
        
        if time >= spike_time:
            for k in range(self.synapses):
                weight = weights[k]
                delay  = k + 1
                ## will fire when current time 
                ## (timeT) >= time of spike + delay otherwise zero
                if time >= (spike_time + delay):
                    output += (weight * self.srfd((time - delay - spike_time)))
                ## else no charge

        ## else none will fire
        return output

    def srfd(self, time):
        ## 100%
        if time >= 0.0:
            return e(time, self.decay) *((1.0/time) - (1.0/self.decay))
        else:
            return 0.0

    def error(self):
        cdef double total = 0.0
        cdef e = 0.0
        for j in range(self.outputs):
            total += (self.output_time[j]-self.desired_time[j]) ** 2.0

        return (total/2.0)



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



