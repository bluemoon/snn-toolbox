# encoding: utf-8
# filename: spikeprop_ng.pyx

## import multiprocessing as mp
## CPU_CORES = mp.cpu_cores() 
import os
import random
import numpy   as np
import cPickle as cp

from spikeprop_ng import *

DECAY       = 7
SYNAPSES    = 16
IPSP        = 1
MAX_TIME    = 50
TIME_STEP   = 0.01
NEG_WEIGHTS = False
MP          = True

QUICKPROP   = False
RPROP       = False

### Converted Properly
##
## - link_out_d
## - link_out
## - e
## - sign
## - srfd


class Math:
    def __init__(self):
        pass

    def e(self, time):
        if time > 0:
            return (time * np.exp( (1.0 - time) / DECAY ))/DECAY
        else:
            return 0

    def sign(self, number):
        if number > 0:
            return 1
        elif number < 0:
            return -1
        else:
            return 0

    def srfd(self, time):
        asrfd = 0
        if time <= 0:
            return asrfd
        else:
            return self.e(time) * ((1.0/time) - (1.0/DECAY))

    def y(self, time, spike, delay):
        return self.e(time-spike-delay)

    def link_out(self, weights, spike, time):
        ## if time >= (spike + delay)
        ## delay_max = SYNAPSES
        ## the delay is 1...16
        ## if time >= (spike + {1...16})
        ## so i need to find the minimum delay size and
        ## start the loop from there
        output = 0.0

        i = int(time-spike)
        #for k in xrange(i):
        for k in xrange(SYNAPSES):
            delay = k+1
            weight = weights[k]
            output += (weight * self.e(time-spike-delay))

        return output

    def link_out_d(self, weights, spike_time, time):
        output = 0.0
        if time >= spike_time:
            for k in xrange(SYNAPSES):
                weight = weights[k]
                delay  = k + 1
                ## will fire when current time 
                ## (timeT) >= time of spike + delay otherwise zero
                if time >= (spike_time + delay):
                    output += (weight * self.srfd((time - delay - spike_time)))
                    ## else no charge
        return output

    def equation_12(self, j):
        return (self.layer.next.time[j]-self.layer.next.desired_time[j]) / \
            (self.equation_12_bottom(j))

    def equation_12_bottom(self, j):
        ot = 0.0
        for i in range(self.layer.prev.size):
            link_out_ = self.link_out_d(self.layer.weights[i, j], 
                            self.layer.prev.time[i], 
                            self.layer.next.time[j])
            if i >= (self.layer.prev.size - IPSP):
                ot = ot - link_out_ 
            else:
                ot = ot + link_out_

        return ot
 
    def equation_17_top(self, i):
        ## the top of equation 17 is from i to j
        ## so in our case it would be from the current layer
        ## (self.layer.prev to self.layer.next) 
        ot = 0.0
        actual = 0.0
        next_layer = self.layers[self.layer_idx+1]
        spike_time = self.layer.next.time[i]
        for j in xrange(next_layer.next.size):
            actual_time = next_layer.next.time[j]
            delta = next_layer.deltas[j]
            link_out_ = self.link_out_d(next_layer.weights[i, j], spike_time, actual_time)
            if i >= (self.layer.next.size - IPSP):
                ot = -link_out_
            else:
                ot = link_out_

            actual = actual + (delta * ot)

        return actual
    
    def equation_17_bottom(self, i):
        ## the bottom of equation 17 is from h to i
        actual = 0.0
        actual_time = self.layer.next.time[i]
        for h in xrange(self.layer.prev.size):
            spike_time = self.layer.prev.time[h]
            ot = self.link_out_d(self.layer.weights[h, i], spike_time, actual_time)
            actual += ot
        
        if i >= (self.layer.next.size - IPSP):
            return -actual
        else:
            return actual

    def equation_17(self, i):
        ## this equation works as follows
        ## we go from i to j in the top equation
        ## we go from h to i in the bottom equation
        ## so we need to go from the current layer
        ##   self.layer.prev as h to self.layer.next as i
        ## and then
        ##   self.layer.next as i to self.layers[self.layer_idx+1].prev as j
        ##
        ## so if  layer[0].prev = 3 which would be h
        ## then   layer[0].next = 5 which would be i
        ## and    layer[1].prev = 5 which would also be i
        ## lastly layer[1].next = 1 which would be j

        actual = self.equation_17_top(i)/self.equation_17_bottom(i)
        return actual
            

    def change(self, actual_time, spike_time, delay, delta):
        return (-self.layer.learning_rate * self.y(actual_time, spike_time, delay) * delta)

    def error_weight_derivative(self, actual_time, spike_time, delay, delta):
        return self.y(actual_time, spike_time, delay) * delta

class neurons:
    def __init__(self, neurons):
        self.neurons      = neurons
        self.time         = np.ndarray((neurons))
        self.desired_time = np.ndarray((neurons))

    @property
    def size(self):
        return self.neurons
    
class layer:
    def __init__(self, previous_neurons, next_neurons):
        previous = previous_neurons.size
        next     = next_neurons.size

        self.prev  = previous_neurons
        self.next  = next_neurons

        self.weights = np.random.rand(previous, next, SYNAPSES) * 10.0
        self.delays  = np.random.rand(previous, next)
        self.deltas  = np.random.rand(next)
        self.derivative   = np.random.rand(previous, next, SYNAPSES)
        self.derive_delta = np.random.rand(previous, next, SYNAPSES)

        self.learning_rate = 1.0
        self.weight_method = 'random1'

        if self.weight_method == 'random1':
            for i in xrange(self.prev.size):
                for h in xrange(self.next.size):
                    r = random.randint(0, 32767) % 10
                    for k in xrange(SYNAPSES):
                        self.weights[i, h, k] = (r + 1)

        elif self.weight_method == 'random2':
            for i in xrange(self.prev.size):
                for h in xrange(self.next.size):
                    for k in xrange(SYNAPSES):
                        r = random.randint(0, 32767) % 10
                        self.weights[i, h, k] = (r + 1)

        elif self.weight_method == 'normalized':
            mu, sigma = 1.11, 0.35
            self.weights = np.random.normal(mu, sigma, size=(previous, next, SYNAPSES))



class modular(Math):
    ## Calculate δj for all outputs according to (12)
    ## For each subsequent layer I = J − {1...2}
    ## Calculate δi for all neurons in I according to (17)
    ## For output layer J , adapt wij by Δwij = −η * Y.i(t.j) * δj
    ## For each subsequent layer I = J − {1...2}
    ## Adapt w.hi by Δw.ij = −η * Y.h(t.i) * δi (18)
    def __init__(self, layers):
        Math.__init__(self)

        self.layers    = layers
        self.threshold = 50
        self.failed    = False
        self.layer     = None
        self.layer_idx = 0

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
        if self.layer_idx >= (len(self.layers) - 1):
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
            
    def quickprop_propagate(self, i, j, k):
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

    def descent_propagate(self, i, j, k):
        if i >= self.layer.prev.size - IPSP:
            change_weight = -self.change(self.actual_time,
                                         self.spike_time,
                                         self.delay, 
                                         self.delta)
        else:
            change_weight = self.change(self.actual_time,
                                        self.spike_time,
                                        self.delay,
                                        self.delta)
        return change_weight

    def backwards_pass(self, input_times, desired_times):
        self.forward_pass(input_times, desired_times)
        if self.fail:
            return False

        ## Go through every layer backwards
        ## doing the following steps:
        for layer_idx in range(len(self.layers)-1, -1, -1):
            self.layer_idx = layer_idx
            self.layer = self.layers[self.layer_idx]
            for i in xrange(self.layer.next.size):
                ##  1) figure out what layer we are on
                ##  2) calculate the delta j or delta i depending on the 
                ##     previous step, if this is the last layer we use
                ##     equation 12, if this is input -> hidden, or hidden to hidden
                ##     then we use equation 17
                if not self.last_layer:
                    self.layer.deltas[i] = self.equation_17(i)

                elif self.last_layer:
                    self.layer.deltas[i] = self.equation_12(i)
                
            ##  3) then we go through all of the next neuron set(j) 
            ##     get the time(actual_time), then go through all the 
            ##     neurons in the previous set(i)
            ##     and get the previously calculated delta
            for j in xrange(self.layer.next.size):
                self.actual_time = self.layer.next.time[j]
                self.delta = self.layer.deltas[j]
                for i in xrange(self.layer.prev.size):
                    ## 4) from there we go through all the synapses(k)
                    for k in xrange(SYNAPSES):
                        self.delay = k+1
                        ## 5) we then get the spike time of the last layer(spike_time)
                        ##    get the last weight(old_weight) and if we are on the last
                        ##    layer
                        ##
                        ##   ∂E
                        ## -------  = ε.ij^k(t-t.i-d.ij^k)δ.{i,j}
                        ## ∂w.ij^k

                        self.spike_time = self.layer.prev.time[i]
                        old_weight = self.layer.weights[i,j,k]
                        delta_weight = self.propagating_routine(i, j, k)
                        new_weight = old_weight + delta_weight
                        
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
        self.layers[0].prev.time           = input_times
        self.layers[-1].next.desired_time  = desired_times
        
        ## XXX: figure out if i should do this forwards
        ## or backwards
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
                        ## get the previous layers spike time
                        if self.first_layer:
                            spike_time = input_times[h]
                        else:
                            spike_time = self.layer.prev.time[h]

                        if time >= spike_time:
                            layer_weights = self.layer.weights[h, i]
                            ot = self.link_out(layer_weights, spike_time, time)
                            if self.last_layer:
                                if i >= self.layer.next.size:
                                    total -= ot
                                else:
                                    total += ot
                            else:
                                total += ot
                    
                    ## now set the next layers spike time to the current time
                    ## XXX: check to see if this can be optimized    
                    self.layer.next.time[i] = time
                    time += TIME_STEP



                    if time >= 50.0:
                        self.failed = True
                        
                        
    @property
    def error(self):
        last_layer = self.layers[-1]
        total = 0.0
        for j in range(last_layer.next.size):
            total += ((last_layer.next.time[j] - last_layer.next.desired_time[j]) ** 2.0)
            
        return (total/2.0)
