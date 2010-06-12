from numpy import *
import random
import scipy
from cy.math import E, math

class connection:
    paramdim = 1
    def __init__(self, in_module, out_module, name=None):
        self.in_module = in_module
        self.out_module = out_module
        self.indim = in_module.outdim
        self.outdim = out_module.indim
        self.dims = [self.indim, self.outdim, 16]
        self._dims = [self.indim,self.outdim]
        
        self.inbuf = self.in_module.outputbuffer
        self.outbuf = self.out_module.inputbuffer

        self.name = name
        self.reset_derivative()

    def reset_derivative(self):
        self._derivs = zeros(self._dims)
        
    def backward(self):
        self._backwardImplementation(self.out_module.inputerror,
                                     self.in_module.outputerror,
                                     self.in_module.outputbuffer)

        
    def forward(self):
        self._forwardImplementation(self.in_module.outputbuffer, self.out_module.inputbuffer)
        
    def _forwardImplementation(self, inbuf, outbuf):
        outbuf = inbuf
    
    def _backwardImplementation(self, outerr, inerr, inbuf):
        #print concatenate(self.in_module._allBuffers())
        #inerr += dot(reshape(self.params, (self.outdim, self.indim)).T, outerr)
        #ds = self.derivs
        #ds += outer(inbuf, outerr).T.flatten()
        #inerr += outerr

        #print "outerr:",outerr
        #print "inbuf:",inbuf
        inerr[:] = outerr[:len(inerr)]
        ds = self._derivs
        ds += outer(inbuf, outerr)
        #print "derivs:",ds
        #print "inerr:",inerr
        
    def whichBuffers(self, paramIndex):
        """Return the index of the input module's output buffer and
        the output module's input buffer for the given weight."""
        return paramIndex % self.inmod.outdim, paramIndex / self.inmod.outdim

class e_in_connection(connection):
    def __init__(self, in_module, out_module, name=None):
        connection.__init__(self, in_module, out_module, name=name)
        self.weights = zeros(self.dims)
        self.math = math()
        self.math.generate_weights((self.outdim, self.indim, 16), self.weights)
        self.deltas = ndarray(self.outdim)
        
        #for i in xrange(self.outdim):
        #    for h in xrange(self.indim):
        #        self.weights[h,i] = random.randint(1,10)
                    
    def backward(self):
        delta_j = self.out_module.connection_tail.deltas
        print delta_j
    
    def forward(self):
        for i in xrange(self.outdim):
            time = 0
            out  = 0
            while (out < 50 and time < 50):
                out = 0
                for h in xrange(self.indim):
                    spike_time = self.inbuf[h]
                    if time >= spike_time:
                        out += self.math.link_out(self.weights[h, i], spike_time, time)
                    else:
                        time = spike_time
                        
                self.outbuf[i] = time
                time += 0.01
                    
    
class e_out_connection(connection):
    def __init__(self, in_module, out_module, name=None):
        connection.__init__(self, in_module, out_module, name=name)
        self.weights = zeros(self.dims)
        self.deltas = ndarray(self.outdim)
        self.math = math()
        for i in xrange(self.outdim):
            for h in xrange(self.indim):
                for k in xrange(16):
                    self.weights[h,i,k] = random.randint(1,10)
                    
        
    def backward(self):
        for j in xrange(self.outdim):
            self.deltas[j] = self.math.equation_12(j, self.out_module.desiredbuffer, self.out_module.outputbuffer, self)
        
    
    def link_out(self, weight, spike, time):
        output = 0
        i = int(time-spike)
        for k in xrange(i):
            delay = k+1
            output += (weight[k] * E(time-spike-delay))

        return output
    
    def forward(self):
        for i in xrange(self.outdim):
            time = 0
            out  = 0
            while (out < 50 and time < 50):
                out = 0
                for h in xrange(self.indim):
                    spike_time = self.inbuf[h]
                    if time >= spike_time:
                        ot = self.link_out(self.weights[h, i], spike_time, time)
                        if (i >= self.indim-1):
                            out=out-ot
                        else:
                            out=out+ot
                    else:
                        time = spike_time
                        
                self.outbuf[i] = time
                time += 0.01
                    
