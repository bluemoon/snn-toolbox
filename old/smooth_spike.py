import numpy as np

LAMBDA   = 7
SYNAPSES = 16



def check_type(f):
    def runner(*args, **kwargs):
        if len([True for x in args if isinstance(x, layer)]) > 0:
            return f(*args, **kwargs)
        else:
            print "Invalid class type"
            
    return runner

class neurons:
    def __init__(self, neurons):
        self.last_spike = [[] for x in xrange(neurons)]

    def last_spike(self, neuron):
        return self.last_spike[neuron][-1]
    
    def spikes(self, neuron):
        print self.last_spike
        return len(self.last_spike[neuron])
    
class layer:
    def __init__(self, in_neurons, out_neurons):
        self.ins  = in_neurons
        self.outs = out_neurons
        self.weights = np.random.rand(self.ins, self.outs) * 10.0
        self.delays  = np.random.rand(self.ins, self.outs)
        self.In  = neurons(self.ins)
        self.Out = neurons(self.outs)

class spikeprop:
    def __init__(self, layers):
        self.layers = layers
    
class spikeprop_smooth:
    def __init__(self):
        pass
    
    def error(self, desired_times):
        sum = 0 
        for j in xrange(self.layer.outs):
            pass
        
    def sigmoidal_gate_0(self, delta, x):
        return self.sigmoidal_gate(0,1,delta,x)
        
    def sigmoidal_gate_prime_0(self, delta, x):
        self.sigmoidal_gate_prime(0,1,delta,x)
        
    def sigmoidal_gate(self, alpha, beta, delta, x):
        if x < 0:
            return alpha
        elif x >= 0 and x <= delta:
            return (beta-alpha)*((6.0*(x/delta)-15.0)* x/delta+10.0)*((x/delta)^2)
        elif x > delta:
            return beta
        
    def sigmoidal_gate_prime(self, alpha, beta, delta, x):
        if x >= 0 and x <= delta:
            return (30.0/delta)*(beta-alpha)*((x/delta - 2.0)*x/delta+1.0)*(x/delta)**2.0
        else:
            return 0
        
    def sigmoid(self, time):
        return (1.0/1.0+np.exp(-LAMBDA*time))
    
    def sigmoid_prime(self, time):
        return (LAMBDA*self.sigmoid(time)*(1.0-self.sigmoid(time)))
    
    def epsilon(self, t):
        delta_0 = 0.5
        return np.exp(-((t-1)**2.0)) * self.sigmoidal_gate_0(delta_0, t)
    
    def epsilon_prime(self, t):
        gamma = 1
        return np.exp(-(t-1)**2.0) * \
               (self.sigmoidal_gate_prime_0(gamma, t) - 2.0*\
                (time-1.0) * self.sigmoidal_gate_0(time))

    def t_js(self, layer):
        for idx, spikes in enumerate(layer.Out.last_spike):
            for spike in spikes:
                pass
            
    
    def excitation(self, time, j):
        ## this is x sub j
        sum = 0
        for i in xrange(self.layer.outs):
            weight = self.layer.weights[j,i]
            delay = self.layer.delays[j,i]
            tau = self.tau(time - delay, self.layer.Out, i)
            sum += self.epsilon(time - delay - tau)
            
        return sum
    
    def excitation_prime(self, time, j):
        sum = 0
        for i in xrange(self.layer.outs):
            weight = self.layer.weights[j,i]
            delay = self.layer.delays[j,i]
            tau = self.tau(time - delay, self.layer.Out, i)
            self.epsilon(time - delay - tau)

    
    def tau(self, time, layer, i):
        sum = 0
        for s in xrange(1, layer.spikes(i)):
            diff = layer.last_spike[i][s] - layer.last_spike[i][s-1]
            sum += diff * self.sigmoid(t-layer.last_spike[i][s])
        return sum

    def tau_prime(self, time, layer, i):
        sum = 0
        for s in xrange(1, layer.spikes(i)):
            diff = layer.last_spike[i][s] - layer.last_spike[i][s-1]
            sum += diff * self.sigmoid(t-layer.last_spike[i][s]) * ()
        return sum
    
    def forward_pass(self, input_times, desired_times, layers):
        for self.layer in layers:
            pass
        

    def train(self, layers):
        for self.layer in layers:
            self.excitation(0,0)
        
if __name__ == "__main__":
    smooth = spikeprop_smooth()
    In  = layer(5,2)
    Out = layer(2,1)

    smooth.train([In, Out])
    assert smooth.epsilon(1) == 1.0
    print smooth.sigmoid(0)
