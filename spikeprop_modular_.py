# encoding: utf-8
# filename: spikeprop_ng.pyx

## import multiprocessing as mp
## CPU_CORES = mp.cpu_cores() 
import os

import numpy   as np
import cPickle as cp

DECAY       = 7
SYNAPSES    = 16
IPSP        = 1
MAX_TIME    = 50
TIME_STEP   = 0.01
NEG_WEIGHTS = False
MP          = True

QUICKPROP   = False
RPROP       = False

class spikeprop_math:
    @staticmethod
    def srfd(time):
        asrfd = 0
        if time <= 0:
            return asrfd
        else:
            return e(time) * ((1.0/time) - (1.0/DECAY))

    @staticmethod
    def y(time, spike, delay):
        return e(time-spike-delay)

    @staticmethod
    def link_out(weights, spike, time):
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

    @staticmethod
    def link_out_d(weights, spike_time, time):
        output = 0.0
        if time >= spike_time:
            for k in range(SYNAPSES):
                weight = weights[k]
                delay  = k + 1
                ## will fire when current time 
                ## (timeT) >= time of spike + delay otherwise zero
                if time >= (spike_time + delay):
                    output += (weight * spikeprop_math.srfd((time - delay - spike_time)))
                    ## else no charge

    def equation_12(self, j):
        return (self.desired_time[j]-self.output_time[j])/(self._e12bottom(j))

    def _e12bottom(self, j):
        ot = 0.0
        for i in range(self.hiddens):
            if i >= (self.hiddens - IPSP):
                ot = ot - self.link_out_d(self.output_weights[j,i], \
                self.hidden_time[i], self.output_time[j])
            else:
                ot = ot + self.link_out_d(self.output_weights[j,i], \
                self.hidden_time[i], self.output_time[j])

        return ot
 
    def equation_17_top(self, i, delta_j):
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
    
    def equation_17_bottom(self, i):
        actual = 0.0
        actual_time = self.hidden_time[i]

        for h in range(self.inputs):
            spike_time = self.input_time[h]
            ot = self.link_out_d(self.hidden_weights[i,h], spike_time, actual_time)
            actual = actual + ot
        
        if i >= (self.hiddens-IPSP):
            return -actual
        else:
            return actual

    def equation_17(self, i):
        actual = self.equation_17_top(i, self.layer.deltas[i])/self.equation_17_bottom(i)
        return actual
            

    def change(self, actual_time, spike_time, delay, delta):
        return (-self.layer.learning_rate * y(actual_time, spike_time, delay) * delta)

class neurons:
    def __init__(self, neurons):
        self.neurons = neurons
        self.time    = np.ndarray((neurons))

    @property
    def size(self):
        return len(self.neurons)
    
class layer:
    def __init__(self, previous, next):
        self.prev  = previous
        self.next  = next
        self.weights = np.random.rand(self.ins, self.outs, SYNAPSES) * 10.0
        self.delays  = np.random.rand(self.ins, self.outs)
        self.deltas  = np.random.rand(self.ins, self.outs)
        self.learning_rate = 1.0
        

class modular(spikeprop_math):
    def __init__(self, layers):
        self.layers    = layers
        self.threshold = 50
        self.failed = False
        self.layer = None
    
    def backwards_pass(self, input_times, desired_times):
        self.forward_pass(input, desired)
        ## Go through every layer backwards
        ## doing the following steps:
        ##  1) figure out what layer we are on
        ##  2) calculate the delta j or delta i depending on the 
        ##     previous step, if this is the last layer we use
        ##     equation 12, if this is input -> hidden, or hidden to hidden
        ##     then we use equation 17
        ##  3) 
        ##    
        for layer_idx in range(len(self.layers)):
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
        return self.error
    

    def forward_pass(self, input_times, desired_times):
        ## The first layer will be the furthest most left
        ## and the last layer will be the furthest most right
        ## 0 and -1 respectively
        self.layers[0].prev.time      = in_times
        self.layers[-1].next.desired  = desired_times
        
        ## XXX: figure out if i should do this forwards
        ## or backwards
        total = 0
        for layer_idx in xrange(len(self.layers)):
            self.layer = self.layers[layer_idx]
            ## we now need to run from layer to layer
            ## first we must go through the next layer size(i)
            ## then we go through the previous layer(h)
            ## get the firing time of the previous layer(h)
            ## and figure out if the sum of all in our current time(time)
            ## passes the threshold value which is calculated with 
            ## spikeprop_math.linkout but because we are a subclass of it
            ## it can be accessed through self
            
            for i in xrange(self.layer.next.size):
                total = 0
                time  = 0
                while (total < self.threshold and time < MAX_TIME):
                    for h in xrange(self.layer.prev.size):
                        ## get the previous layers spike time
                        spike_time = input_times[h]
                        if time >= spike_time:
                            layer_weights = self.layer.weights[h,i]
                            total += self.link_out(layer_weights, spike_time, time)
                            
                        time += TIME_STEP

                    ## now set the next layers spike time to the current time
                    ## XXX: check to see if this can be optimized    
                    self.layer.next.time[i] = time

                    if time >= 50.0:
                        self.failed = True
                        
                        
    @property
    def error(self):
        last_layer = self.layers[-1]
        total = 0.0
        for j in range(last_layer.next.size):
            total += ((last_layer.next.time[j]-last_layer.next.desired[j]) ** 2.0)
            
        return (total/2.0)
