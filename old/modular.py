# encoding: utf-8

from conf           import *
from structure      import *
from spikeprop_math import *
from debug          import *

if WEAVE:
    from scipy.weave import *
    from numpy.distutils.system_info import get_info
    from os.path import join, split
if MP:
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

class modular(Math):
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
    def __init__(self, layers):
        self.layers    = layers
        self.threshold = 50
        self.failed    = False
        self.layer     = None
        self.layer_idx = 0
        
        self.output_layer = self.layers[-1]
        self.input_layer  = self.layers[0]

        self.propagating_type = 'descent'
        self.propagating_routine = getattr(self, self.propagating_type + '_propagate')
        
    @property
    def fail(self):
        return self.failed

    @property
    def neg_weights(self):
        return NEG_WEIGHTS

    @property
    def last_layer(self):
        if self.layer_idx == (len(self.layers) - 1):
            last_layer = True
        else:
            last_layer = False

        return last_layer

    @property
    def first_layer(self):
        if self.layer_idx == 0:
            first_layer = True
        else:
            first_layer = False
        return first_layer

    @property
    def hidden_layer(self):
        if not self.first_layer and not self.last_layer:
            return True
        else:
            return False
            
    def quickprop_propagate(self, i, j, k, actual, spike, delay, delta):
        momentum = 0

        double_prime = self.layer.derivative[i,j,k]
        prime = self.error_weight_derivative(self.actual_time, 
                                             self.spike_time, 
                                             self.delay, 
                                             self.delta)

        if self.sign(prime) == self.sign(double_prime):
            value =  -self.layer.learning_rate * prime + \
                (momentum * self.layer.derive_delta[i,j,k])
        else:
            value = (prime / (double_prime - prime)) * self.layer.derive_delta[i, j, k]
        
        self.layer.derive_delta[i, j, k] = value
        self.layer.derivative[i, j, k]   = prime
        return value

    def descent_propagate(self, i, j, k, actual, spike, delay, delta):
        if not self.last_layer:
            layer = self.layer.next.size
            m = j
        else:
            layer = self.layer.prev.size
            m = i

        if m >= (layer - IPSP):
            #debug("IPSP Call: neuron: %d layer: %d layer size: %d" % (m, self.layer_idx, layer))
            delta_weight = -self.change(actual, spike, delay, delta)
        else:
            delta_weight = self.change(actual, spike, delay, delta)
    
        return delta_weight

    def _delta_j(self):
        self.layer = self.layers[-1]
        self.layer_idx = len(self.layers)-1
        for j in xrange(self.layer.next.size):
            self.layer.deltas[j] = self.delta_j(j)
            
    def _delta_i(self):
        for layer_idx in xrange(len(self.layers)-1, -1, -1):
            self.layer_idx = layer_idx
            self.layer = self.layers[self.layer_idx]
            if not self.last_layer:
                for i in xrange(self.layer.next.size):
                    self.layer.deltas[i] = self.delta_i(i)
                

    def backwards_pass(self, input_times, desired_times):
        self.forward_pass(input_times, desired_times)
        if self.fail:
            return False
        
        self._delta_j()
        self._delta_i()
        ## Go through every layer backwards
        ## doing the following steps:
        for layer_idx in xrange(len(self.layers)-1, -1, -1):
            ##  1) figure out what layer we are on
            self.layer_idx = layer_idx
            self.layer = self.layers[self.layer_idx]
            ##  2) calculate the delta j or delta i depending on the 
            ##     previous step, if this is the last layer we use
            ##     equation 12, if this is input -> hidden, or hidden to hidden
            ##     then we use equation 17

            ##  3) then we go through all of the next neuron set(j) 
            ##     get the time(actual_time), then go through all the 
            ##     neurons in the previous set(i)
            ##     and get the previously calculated delta
            for j in xrange(self.layer.next.size):
                actual_time = self.layer.next.time[j]
                delta = self.layer.deltas[j]
                for i in xrange(self.layer.prev.size):
                    ## 4) from there we go through all the synapses(k)
                    spike_time = self.layer.prev.time[i]
                    for k in xrange(SYNAPSES):
                        ## 5) we then get the spike time of the last layer(spike_time)
                        ##    get the last weight(old_weight) and if we are on the last
                        ##    layer
                        delay = k + 1
                        old_weight = self.layer.weights[i, j, k]
                        delta_weight = self.propagating_routine(i, j, k, actual_time, spike_time, delay, delta)                        

                        new_weight = old_weight + delta_weight
                        #new_weight = corefunc.add(old_weight, delta_weight)
                        
                        if self.neg_weights:
                            self.layer.weights[i, j, k] = new_weight
                        else:
                            if new_weight >= 0.0:
                                self.layer.weights[i, j, k] = new_weight
                            else:
                                self.layer.weights[i, j, k] = 0.0

        return self.error
    

    def forward_pass(self, input_times, desired_times):
        ## The first layer will be the furthest most left
        ## and the last layer will be the furthest most right
        ## 0 and -1 respectively
        self.input_layer.prev.time           = input_times
        self.output_layer.next.desired_time  = desired_times
        
        time  = 0
        total = 0
        for layer_idx in xrange(len(self.layers)):
            self.layer = self.layers[layer_idx]
            self.layer_idx = layer_idx
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
                    total = 0
                    for h in xrange(self.layer.prev.size):
                        spike_time = self.layer.prev.time[h]
                        ## when the time is past the spike time
                        if time >= spike_time:
                            ot = self.excitation(self.layer.weights[h, i], spike_time, time)
                            if self.last_layer:
                                if h >= (self.layer.prev.size - IPSP):
                                    #debug("IPSP Call: neuron: %d layer: %d layer size: %d" % (h, self.layer_idx, self.layer.prev.size))
                                    total -= ot
                                else:
                                    total += ot
                            else:
                                total += ot
                                
                    self.layer.next.time[i] = time
                    ## now set the next layers spike time to the current time
                    ## XXX: check to see if this can be optimized    
                    time += TIME_STEP
                    
                
                if time >= 50.0:
                    self.failed = True
                    break

        return self.error
                        
    @property
    def error(self):
        last_layer = self.layers[-1]
        total = 0.0
        for j in range(last_layer.next.size):
            total += (last_layer.next.time[j] - last_layer.next.desired_time[j]) ** 2.0
            
        return (total/2.0)



if __name__ == "__main__":
    import doctest
    doctest.testmod()
