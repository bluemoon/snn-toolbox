from cy.math import E
from circuits import Event, Component, Manager, Debugger, future

from multiprocessing import Lock
from defers import *

import numpy  as np
import random
import time as time_

MAX_TIME = 50

class Ev(Event): 
    def __init__(self, *args, **kwargs):
        super(Ev, self).__init__(*args, **kwargs)

def xor(which):
    if which == 0:
        input = np.array([0.0, 0.1, 0.1])
        desired = np.array([16.0])
    elif which == 1:
        input = np.array([0.0, 0.1, 6.0])
        desired = np.array([10.0])
    elif which == 2:
        input = np.array([0.0, 6.0, 0.1])
        desired = np.array([10.0])
    elif which == 3:
        input = np.array([0.0, 6.0, 6.0])
        desired = np.array([16.0])
    else:
        input = np.array([0.0, 6.0, 6.0])
        desired = np.array([16.0])

    return input, desired

class fire_neuron(Ev):pass
class forward_while_event(Ev):pass
class forward_sub1(Ev):pass
class forward_pass(Ev):pass
class increment(Ev):pass
class decrement(Ev):pass
class neuron:
    def __init__(self, size):
        self.size = size
        self.time = np.ndarray((size))
        
class layer:
    def __init__(self, previous, next):
        self.previous = neuron(previous)
        self.next = neuron(next)
        self.weights = np.ndarray((previous, next, 16))
        self.deltas = np.ndarray(next)
        for i in xrange(self.previous.size):
            for h in xrange(self.next.size):
                for k in xrange(16):
                    r = random.randint(1, 10)
                    self.weights[i, h, k] = r
                    
    def __repr__(self):
        return '<layer prev size:%d next size: %d>' % (self.previous.size, self.next.size)
    
class input_layer(layer):pass
class hidden_layer(layer):pass
class output_layer(layer):pass
class get_value(layer):pass


            
class network_container(Component):
    def __init__(self):
        super(network_container, self).__init__()
        self.threshold = 50
        self.lock = Lock()
        self.layers = []

    def __iadd__(self, other):
        if isinstance(other, layer):
            self.layers.append(other)
            
        return self
    
    def decrement(self, shared, amount):
        self.lock.acquire()
        value = shared - amount
        self.lock.release()
        return value

    def increment(self, shared, amount):
        self.lock.acquire()
        value = shared + amount
        self.lock.release()
        return value
    
    @future()    
    def forward_sub1(self, weights, time, spike):
        target = int(time-spike)
        ot = 0
        for k in xrange(target):
            delay = k + 1
            weight = weights[k]
            ot += (weight * E(time-spike-delay))

        return ot
    
    @future()
    def forward_while_event(self, total, layer, time, i):
        for h in range(layer.previous.size):
            spike_time = layer.previous.time[h]            
            if time >= spike_time:
                weights = layer.weights[h, i]
                ot = self.push(forward_sub1(weights, time, spike_time))
                if h >= (layer.previous.size - 1):
                    self.push(decrement(total, ot))
                else:
                    self.push(increment(total, ot))
                    
        return total
        
    
    def forward_pass(self, input, desired):
        self.layers[0].previous.time  = input
        self.layers[-1].next.time = desired
        for idx in xrange(len(self.layers)):
            layer = self.layers[idx]
            for i in range(layer.next.size):
                total = 0
                time  = 0
                while (total < self.threshold and time < MAX_TIME):        
                    total = 0
                    total = self.push(forward_while_event(total, layer, time, i))
                    time += 0.01
                            
                

    
