import numpy as np

LAMBDA   = 7
SYNAPSES = 16

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

        return len(self.last_spike[neuron])
    
    def add_spike(self, neuron, time):
        self.last_spike[neuron].append(time)
        
    @property
    def size(self):
        return len(self.last_spike)
    
class layer:
    def __init__(self, previous_count, next_count):
        self.previous  = neurons(previous_count)
        self.next      = neurons(next_count)
        self.weights = np.random.rand(self.previous.size, self.next.size) * 10.0
        self.delays  = np.random.rand(self.previous.size, self.next.size)


class spikeprop_smooth:
    def __init__(self, layers):
        self.layers = layers
        self.c = 0.0
        
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
            return (beta-alpha)*((6.0*(x/delta)-15.0)* x/delta+10.0)*((x/delta)**2.0)
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
                (t-1.0) * self.sigmoidal_gate_0(gamma, t))

    def t_js(self, layer):
        for idx, spikes in enumerate(layer.next.last_spike):
            for spike in spikes:
                pass
            
    
    def excitation(self, time, j):
        ## this is x sub j
        sum = 0
        for i in xrange(self.layer.next.size):
            weight = self.layer.weights[j,i]
            delay = self.layer.delays[j,i]
            tau = self.tau(time - delay, self.layer.next, i)
            sum += self.epsilon(time - delay - tau)
            
        return sum
    
    def excitation_prime(self, time, j):
        sum = 0
        for i in xrange(self.layer.next.size):
            weight = self.layer.weights[j,i]
            delay = self.layer.delays[j,i]
            tau = self.tau(time - delay, self.layer.next, i)
            ep = self.epsilon_prime(time - delay - tau) * (1.0 - self.tau_prime(t-delay, i))
            sum += weight * ep

    
    def tau(self, time, layer, i):
        sum = 0
        for s in xrange(1, layer.spikes(i)):
            diff = layer.last_spike[i][s] - layer.last_spike[i][s-1]
            sum += diff * self.sigmoid(t-layer.last_spike[i][s])
        return sum

    def tau_prime(self, time, i):
        sum = 0
        for s in xrange(1, self.layer.spikes(i)):
            diff = self.layer.last_spike[i][s] - self.layer.last_spike[i][s-1]
            sum += diff * self.sigmoid(t-self.layer.last_spike[i][s]) * (1 - self.sigmoid(t-self.layer.last_spike[i][s])) 
        return LAMBDA * sum
    
    #def forward_pass(self, input_times, desired_times):
    #    time = 1.0
    #    for self.layer in self.layers:
            #self.tau(time, self.layer, i)
            #self.excitation()

    def train(self, input, desired):
        T = 40
        for idx, self.layer in enumerate(self.layers):
            for i in range(self.layer.previous.size):
                t = 0
                while t < T:
                    last = self.excitation(t, i)
                    print last
                    print self.excitation_prime(t, i)
                    t += 0.1
                #else:
                    #last = self.excitation()
                
        
if __name__ == "__main__":
    In  = layer(5, 2)
    Out = layer(2, 1)
    smooth = spikeprop_smooth([In, Out])
    input, desired = xor(1)
    smooth.train(input, desired)

    assert smooth.epsilon(1) == 1.0
    #print smooth.sigmoid(0)
