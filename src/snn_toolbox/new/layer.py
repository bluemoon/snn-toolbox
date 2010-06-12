from cy.math import math, E
import numpy as np
import random


SYNAPSES = 16

class layer_base:
    def __init__(self, previous_neurons, next_neurons, shape): 
        previous = previous_neurons.size
        next     = next_neurons.size

        self.prev  = previous_neurons
        self.next  = next_neurons

        self.prev_dim = previous
        self.next_dim = next
        
        self.weights = np.ndarray(shape)
        self.deltas  = np.ndarray((next))
        self.derivative   = np.ndarray(shape) 
        self.weight_delta = np.ndarray(shape) 
        self.learning_rate = 1.0
        self.threshold = 50
        self.weight_method = 'random2'
        
        if self.weight_method == 'random1':
            for i in xrange(self.prev.size):
                for h in xrange(self.next.size):
                    r = random.randint(1, 10)
                    #r = c_rand() % 10.0
                    for k in xrange(SYNAPSES):
                        self.weights[i, h, k] = r

        elif self.weight_method == 'random2':
            for i in xrange(self.prev.size):
                for h in xrange(self.next.size):
                    for k in xrange(SYNAPSES):
                        r = random.randint(1, 10)
                        self.weights[i, h, k] = r

        elif self.weight_method == 'normalized':
            mu, sigma = 1, 3
            self.weights = np.random.normal(mu, sigma, size=(previous, next, SYNAPSES))

    def __repr__(self):
        return '<layer prev size:%d next size: %d>' % (self.prev.size, self.next.size)

    def forward_implementation(self):
        pass
    
    def backward_implementation(self):
        pass
     
    def forward(self, *args):
        self.forward_implementation()
        
    def backward(self, *args):
        self.backward_implementation()
    
    def activate(self, input):
        pass
    
class layer(layer_base):
    def __init__(self, previous_neurons, next_neurons):
        shape = (previous_neurons.size, next_neurons.size, SYNAPSES)
        layer_base.__init__(self, previous_neurons, next_neurons, shape)


class e_out(layer_base):
    def __init__(self, previous_neurons, next_neurons):
        self.shape = (previous_neurons.size, next_neurons.size, SYNAPSES)
        layer_base.__init__(self, previous_neurons, next_neurons, self.shape)
        self.learning_rate = 1.0
        self.math = math()

    def backward(self, *args):
        for j in xrange(self.next.size):
            self.deltas[j] = self.math.equation_12(j, self.next.desired_time, self.next.time, self)

        for j in xrange(self.next.size):
            actual_time = self.next.time[j]
            delta = self.deltas[j]
            for i in xrange(self.prev.size):
                spike_time = self.prev.time[i]
                for k in xrange(16):
                    delay = k + 1
                    weight = self.weights[i,j,k]
                    if i >= self.prev.size-1:
                        delta_weight = -self.learning_rate * self.math.change(actual_time, spike_time, delay, delta)
                    else:
                        delta_weight = self.learning_rate * self.math.change(actual_time, spike_time, delay, delta)

                    new_weight = weight + delta_weight
                    if new_weight >= 0.0:
                        self.weights[i,j,k] = new_weight
                    else:
                        self.weights[i,j,k] = 0
            
    def forward(self):
        for i in xrange(self.next.size):
            time = 0
            out  = 0
            while (out < self.threshold and time < 50):
                out = 0
                for h in xrange(self.prev.size):
                    spike_time = self.prev.time[h]
                    if time >= spike_time:
                        ot = self.math.link_out(self.weights[h, i], spike_time, time)
                        if (i >= self.prev.size-1):
                            out -= ot
                        else:
                            out += ot
                        
                self.next.time[i] = time
                time += 0.01
                
class e_hidden(layer_base):
    def __init__(self, previous_neurons, next_neurons):
        self.shape = (previous_neurons.size, next_neurons.size, SYNAPSES)
        layer_base.__init__(self, previous_neurons, next_neurons, self.shape)
        self.math = math()
        self.learning_rate = 1.0

    def backward(self, *args):
        deltas_ = self.layers[-1].deltas
        for i in xrange(self.next.size):
            self.deltas[i] = self.math.equation_17(i, deltas_, self)


        for i in xrange(self.next.size):
            actual_time = self.next.time[i]
            delta = self.deltas[i]
            for h in xrange(self.prev.size):
                spike_time = self.prev.time[h]
                for k in xrange(16):
                    delay = k + 1
                    weight = self.weights[h,i,k]
                    if i >= self.next.size-1:
                        delta_weight = -self.learning_rate * self.math.change(actual_time, spike_time, delay, delta)
                    else:
                        delta_weight = self.learning_rate * self.math.change(actual_time, spike_time, delay, delta)

                    new_weight = weight + delta_weight
                    if new_weight >= 0.0:
                        self.weights[h,i,k] = new_weight
                    else:
                        self.weights[h,i,k] = 0
                        
    def forward(self):
        for i in xrange(self.next.size):
            time = 0
            out  = 0
            while (out < self.threshold and time < 50):
                out = 0
                for h in xrange(self.prev.size):
                    spike_time = self.prev.time[h]
                    if time >= spike_time:
                        out += self.math.link_out(self.weights[h, i], spike_time, time)
                        
                self.next.time[i] = time
                time += 0.01
                
            if time >= 50:
                raise Exception
