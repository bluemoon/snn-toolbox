# encoding: utf-8
## import multiprocessing as mp
## CPU_CORES = mp.cpu_cores()

WEAVE       = True
DECAY       = 7
SYNAPSES    = 16
IPSP        = 1
MAX_TIME    = 50
TIME_STEP   = 0.01
NEG_WEIGHTS = False
MP          = True

QUICKPROP   = False
RPROP       = False

if WEAVE:
    from scipy.weave import *
    from numpy.distutils.system_info import get_info
    from os.path import join, split
if MP:
    from multiprocessing import Process, Value, Array, Pool, Queue, Lock
    
import os
import time as Time
import random
import numpy   as np
import cPickle as cp
from spikeprop_math import *

def e_sub_(queue, result, lock):
    Continue = True
    while Continue:
        try:
            #print("Reading from queue...")
            ret = queue.get(True, 1)
            if ret == "quit":
                print("Got quit message")
                Continue = False
            else:
                k = ret[0]
                delay = k + 1
                weight = ret[1][k]
                
                lock.acquire()
                value = (weight * Math.e(ret[2] - ret[3] - delay))
                result.value += value
                lock.release()
                
        except Exception:
            pass


def e_sub(k, time, spike, weights):
    delay = k + 1
    weight = weights[k]
    return (weight * Math.e(time - spike - delay))
        

class neurons:
    time = None
    desired_time = None
    
    def __init__(self, neurons):
        self.neurons      = neurons
        self.time         = np.ndarray((neurons))
        self.desired_time = np.ndarray((neurons))

        
    @property
    def size(self):
        return self.neurons
    
class layer:
    weights = None
    deltas  = None
    derivative = None
    derive_delta = None
    prev = None
    next = None
    
    def __init__(self, previous_neurons, next_neurons):
        previous = previous_neurons.size
        next     = next_neurons.size

        self.prev  = previous_neurons
        self.next  = next_neurons

        self.weights = np.random.rand(previous, next, SYNAPSES) * 10.0
        #self.delays  = np.random.rand(previous, next)
        self.deltas  = np.zeros(next)
        self.derivative   = np.random.rand(previous, next, SYNAPSES)
        self.derive_delta = np.random.rand(previous, next, SYNAPSES)

        self._learning_rate = 1.0

        self.weight_method = 'random1'
        
        
        if self.weight_method == 'random1':
            for i in xrange(self.prev.size):
                for h in xrange(self.next.size):
                    r = random.randint(0, 255) % 10.0
                    for k in xrange(SYNAPSES):
                        self.weights[i, h, k] = (r + 1)

        elif self.weight_method == 'random2':
            for i in xrange(self.prev.size):
                for h in xrange(self.next.size):
                    for k in xrange(SYNAPSES):
                        r = random.randint(0, 32767) % 10
                        self.weights[i, h, k] = (r + 1)

        elif self.weight_method == 'normalized':
            mu, sigma = 5, 1
            self.weights = np.random.normal(mu, sigma, size=(previous, next, SYNAPSES))

    @property
    def learning_rate(self):
        return self._learning_rate

class modular(Math):
    ## H (input), I (hidden) and J (output)
    ## 1) Firing time is x.j ≥ v , progressing from the
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
    def __init__(self, layers):
        self.layers    = layers
        self.threshold = 50
        self.failed    = False
        self.layer     = None
        self.layer_idx = 0
        
        self.output_layer = layers[-1]
        self.input_layer  = layers[0]

        self.propagating_type = 'descent'
        self.propagating_routine = getattr(self, self.propagating_type + '_propagate')
        #self.pool = Pool(2)
        #self.queue   = Queue()
        #self.results = Queue()
        #self.lock  = Lock()
        #self.shared_result = Value('d', 0)
        #self.e_workers = [Process(target=e_sub_, args=(self.queue, self.shared_result, self.lock)) for i in range(4)]
        #[p.start() for p in self.e_workers]
        
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
        change_weight = self.change(self.actual_time,
                                     self.spike_time,
                                     self.delay, 
                                     self.delta)

        if self.last_layer and i >= (self.layer.prev.size - IPSP):
            return -change_weight
        elif self.last_layer:
            return change_weight
            
        if not self.last_layer and i >= (self.layer.next.size - IPSP):
            return -change_weight
        else:
            return change_weight

        return change_weight

    def _delta_j(self):
        for j in xrange(self.output_layer.next.size):
            self.output_layer.deltas[j] = self.delta_j(j)
            
    def _delta_i(self):
        if not self.last_layer:
            for i in xrange(self.layer.next.size):
                self.layer.deltas[i] = self.delta_i(i)
                
    @profile   
    def backwards_pass(self, input_times, desired_times):
        self.forward_pass(input_times, desired_times)
        if self.fail:
            return False
        
        self._delta_j()
        ## Go through every layer backwards
        ## doing the following steps:
        for layer_idx in xrange(len(self.layers)-1, -1, -1):
            self.layer_idx = layer_idx
            self.layer = self.layers[self.layer_idx]
            self._delta_i()
            ##  1) figure out what layer we are on
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
                #print layer_idx, self.layer.deltas
                for i in xrange(self.layer.prev.size):
                    ## 4) from there we go through all the synapses(k)
                    spike_time = self.layer.prev.time[i]
                    for k in xrange(SYNAPSES):
                        ## 5) we then get the spike time of the last layer(spike_time)
                        ##    get the last weight(old_weight) and if we are on the last
                        ##    layer

                        ##   ∂E
                        ## -------  = ε.ij^k(t-t.i-d.ij^k)δ.{i,j}
                        ## ∂w.ij^k
                        
                        delay = k + 1
                        old_weight = self.layer.weights[i, j, k]
                        ## delta_weight = self.propagating_routine(i, j, k)
                        if not self.last_layer:
                            layer = self.layer.next.size
                        else:
                            layer = self.layer.prev.size

                        if i >= layer - IPSP:
                            delta_weight = -self.change(actual_time, spike_time, delay, delta)
                        else:
                            delta_weight = self.change(actual_time, spike_time, delay, delta)
                        
                        new_weight = old_weight + delta_weight
                        
                        if self.neg_weights:
                            self.layer.weights[i, j, k] = new_weight
                        else:
                            if new_weight >= 0.0:
                                self.layer.weights[i, j, k] = new_weight
                            else:
                                self.layer.weights[i, j, k] = 0.0



        return self.error
    
    @profile
    def forward_pass(self, input_times, desired_times):
        ## The first layer will be the furthest most left
        ## and the last layer will be the furthest most right
        ## 0 and -1 respectively
        self.input_layer.prev.time           = input_times
        self.output_layer.next.desired_time  = desired_times
        
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
                        spike_time = self.layer.prev.time[h]
                        ## when the time is past the spike time
                        if time >= spike_time:
                            ot = self.excitation(self.layer.weights[h, i], spike_time, time)
                            if self.last_layer:
                                if i >= (self.layer.prev.size - IPSP):
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

        return self.error
                        
    @property
    def error(self):
        last_layer = self.layers[-1]
        total = 0.0
        for j in range(last_layer.next.size):
            total += (last_layer.next.time[j] - last_layer.next.desired_time[j]) ** 2
            
        return (total * 0.5)



if __name__ == "__main__":
    import doctest
    doctest.testmod()
