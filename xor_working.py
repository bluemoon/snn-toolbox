import matplotlib
matplotlib.use('GTKAgg')
from brian.equations import *
from brian.directcontrol import *
from brian import *
from scipy import *
from time import time
from random import randrange
import sympy



########### Parameters
on          = 0
off         = 6
run_time    = 500*ms
tau         = 7*ms
eqs_n = """
#du/dt=-u/tau : 1
dv/dt = (-w-v)/(10*ms) : volt # the membrane equation
dw/dt = -w/(30*ms) : volt # the adaptation current

"""


x1 = [off, off, on, on]
x2 = [off, on, off, on]

def nextspike():
    ## this is the generator for the input data
    base = 0*ms
    while True:
        r = randrange(0, len(x1))
        yield ((x1[r]*ms+base), (x2[r]*ms+base))
        base += 16*ms

def xor_diff(x):
    ## this is the comparator for the input data
    bools  = []
    target = []
    t      = []
    for idx, y in enumerate(x):
        base = idx*float(16*ms)
        t.append(base)

        if y > 0:
            bools.append(True)
            target.append(base+float(10*ms))
        if y < 0:
            bools.append(True)
            target.append(base+float(10*ms))
        if y == 0:
            bools.append(False)
            target.append(base+float(16*ms))

    return bools, target, t


class SpikeInputs(NeuronGroup):
    def __init__(self, neurons, spiketimes, clock=None, period=None):
        clock = guess_clock(clock)
        thresh = SpikeThresh(neurons, spiketimes, period=period)
        NeuronGroup.__init__(self, neurons, model=LazyStateUpdater(),threshold=thresh,clock=clock)
    def reinit(self):
        super(SpikeThresh, self).reinit()
        self._threshold.reinit()        
    def __repr__(self):
        return "MultipleSpikeGeneratorGroup"

class SpikeThresh(Threshold):
    def __init__(self, neurons, spiketimes, period=None):
        self.neurons = neurons
        self.set_spike_times(spiketimes, period=period)
        self.spiketimeiter = self.spiketimes()
        self.nextspiketime = self.spiketimes().next()
        self.fired = []

    def reinit(self):
        self.curperiod = -1

    def set_spike_times(self,spiketimes,period=None):
        self.spiketimes = spiketimes
        self.period = period
        self.reinit()

    def __call__(self, P):
        firing = zeros(self.neurons)
        t = P.clock.t
        for idx, spike in enumerate(self.nextspiketime):
            if spike is not None and float(spike) <= float(t):
                if idx not in self.fired:
                    firing[idx] = 1
                    self.fired.append(idx)


        if len(self.fired) == self.neurons and (float(t) % 16):
            self.nextspiketime = self.spiketimeiter.next()
            self.fired = []

        return where(firing)[0]


class bpnn(NetworkOperation):
    def __init__(self, connection, layer, clock=clock):
        NetworkOperation.__init__(self, lambda:None, clock=clock)
        self.connection = connection
        self.C = connection[-1]

        self.layer = layer
        ## Monitors
        self.d_mon   = SpikeMonitor(layer.data)
        #self.h_mon   = SpikeMonitor(layer.hidden)
        self.o_mon   = SpikeMonitor(layer.output)
        
        self.h_mon   = SpikeMonitor(connection[-2].source)
        self.i_mon   = SpikeMonitor(connection[-1].source)
        self.j_mon   = SpikeMonitor(connection[-1].target)
        self.contained_objects += [self.d_mon, self.h_mon, self.o_mon, self.i_mon, self.j_mon]
        ## Constants
        self.tau         = 7
        self.learn_rate  = 0.05


    def post_d(self, j):
        target = xor_diff(self.d_mon.spiketimes[0] - self.d_mon.spiketimes[1])[1]
        return target[self.time]

    def post_a(self, j):
        return self.j_mon[j][self.time]

    def pre_a(self, i):
        return self.i_mon[i][self.time]

    def h(self, h):
        return self.h_mon[h][self.time]

    def delay(self, i, j):
        return float(self.C.delay[i,j])

    def to_ms(self, num):
        return num*ms

    def from_ms(self, num):
        ## This is a major hack... the reason for this is the data is stored
        ## with a "unit" ms/mv etc. and when converted to a float it happens
        ## to be a really tiny decimal and is not useful in the calculations
        return float(num)*10000

    def _epsilon(self, t):
        ## Formula [3] SpikeProp, Bohte. Et al.
        return (t/self.tau)*exp(1-(t/self.tau))

    def _depsilon(self, t):
        ## The derivative of _epsilon
        return -(exp(1-t/self.tau)*t)/self.tau**2 + (exp(1-t/self.tau)/self.tau)

    def _dy(self, dy, dt, delay):
        return self._depsilon(self.from_ms(dt-dy-delay))

    def _y(self, t, ti, delay):
        ## Formula [2] SpikeProp, Bohte.
        ## y.i(t) = epsilon(t-t.i)
        diff = self.from_ms(t-ti-delay)
        return self._epsilon(diff)

    def _gamma_j(self, j):
        ## Formula [12] SpikeProp, Bohte.
        ## 
        ##    t[j][desired] - t[j][actual]
        ##   ---------------------------------------------------
        ##    sum w.ij(pd y.i(t[j][actual])/pd t[j][actual])
        ##  i subset j 
        sum = 0.0  
        ## desired - actual
        top = self.from_ms(self.post_d(j) - self.post_a(j))
        ## from the last layer to this one
        for i in xrange(self.C.W.shape[0]):
            ## sum all the connection weights 
            ## times the derivative of epsilon
            sum += self.C.W[i,j] * self._dy(self.pre_a(i), self.post_a(j), self.delay(i,j)) 
            
        return top/sum

    def _gamma_i(self, i):
        sum = 0.0
        
        ## Set self.C to the last layer
        self.C = self.connection[-1]
        for j in xrange(len(self.j_mon.spiketimes)):
            sum += self._gamma_j(j) * self.connection[-1].W[i,j] * self._dy(self.post_a(j), self.pre_a(i), self.delay(i,j)) 

        self.C = self.connection[-2]

        conn_sum = 0.0
        for h in xrange(len(self.h_mon.spiketimes)):
            conn_sum += self.C.W[h, i] * self._dy(self.pre_a(i), self.h(h), self.delay(h,i)) 

        return sum/conn_sum

    def _error_(self):
        sum = 0.0
        for j in xrange(self.C.W.shape[1]):
            sum += self.from_ms(self.post_a(j) - self.post_d(j))
        return 0.5*(sum**2)

    def _set_time(self):
        ## there are 2 pre-useful time cycles we dont need, so start 
        ## off with -2
        self.time =  int(floor((float(self.clock.t)*1000)/16.0))-2

    def __call__(self):
        ## Get the differences between spike times, 
        ## this will give us the target value we need
        
        self._set_time()
        print "time:", self.time

        if self.time > 1:
            print "MSE rate:", self._error_()
            ## frome the h->o layer
            for i in xrange(self.C.W.shape[0]):
                for j in xrange(self.C.W.shape[1]):
                    dw = -self.learn_rate*self._gamma_j(j) * \
                        self._y(self.post_a(j), self.pre_a(i), self.delay(i,j))

                    self.C.W[i,j] += float(dw*mV)
                    #print "dw i,j:", dw

            ## run the backprop through the i->h layer
            self.C = self.connection[-2]
            for h in xrange(self.C.W.shape[0]):
                for i in xrange(self.C.W.shape[1]):
                    #print "gamma i:", self._gamma_i(i)
                    dw = -self.learn_rate*self._gamma_i(i) * \
                        self._y(self.h(h), self.pre_a(i), self.delay(h,i))

                    self.C.W[h,i] += float(dw*mV)
        
            self.C = self.connection[-1]

        
        
class layers:
    ## container class
    input  = None
    hidden = None
    output = None
    data   = None
    
    def __init__(self, data, input, hidden, output):
        self.input  = input
        self.hidden = hidden
        self.output = output
        self.data   = data


class connections:
    ## container class
    data_to_input    = None
    input_to_hidden  = None
    hidden_to_output = None

    def __init__(self, data_to_input, input_to_hidden, hidden_to_output):
        self.input_to_hidden  = input_to_hidden
        self.hidden_to_output = hidden_to_output
        self.data_to_input    = data_to_input

    def __getitem__(self, n):
        return [self.data_to_input, self.input_to_hidden, self.hidden_to_output][n]


net = Network()
def backPropagate_setup(input_neurons, hidden_neurons, output_neurons):
    slow_clock = Clock(dt=16*ms) 
    fast_clock = Clock(dt=1*ms)
    #neurons = NeuronGroup(input_neurons+hidden_neurons+output_neurons, model=eqs_n, threshold=Vcut, reset="vm=Vr;w+=b", freeze=True, clock=fast_clock)
    neurons = NeuronGroup(input_neurons+hidden_neurons+output_neurons,
    model=eqs_n, threshold=20*mV, reset='''v  = 0*mV;w += 3*mV ''', 
    refractory=5*msecond, freeze=True, clock=fast_clock)
    
    input  = neurons[0:input_neurons]
    hidden = neurons[input_neurons:input_neurons+hidden_neurons]
    output = neurons[input_neurons+hidden_neurons:]
    ## Inputs
    data = SpikeInputs(2, nextspike, clock=fast_clock)
    net.add(data)
    layer = layers(data, input, hidden, output)
    ## setup clocks, one slow dt=16ms and one fast dt=1ms
    ## the reasons for this are for the time delays in
    ## the connections. which vary from 0-16ms

    d_to_i = Connection(data, layer.input)
    d_to_i.connect_one_to_one(data, layer.input)
    ## input to hidden
    W = rand(len(layer.input),len(layer.hidden))
    #W = clip(W, 0, artificial_wmax)
    i_to_h = DelayConnection(layer.input,  layer.hidden, structure='dense', max_delay=5.0*ms)
    i_to_h.connect(layer.input, layer.hidden, W)
    ## hidden to output
    W=rand(len(layer.hidden),len(layer.output))
    #W = clip(W, 0, artificial_wmax)
    h_to_o = DelayConnection(layer.hidden, layer.output, structure='dense', max_delay=5.0*ms)
    h_to_o.connect(layer.hidden, layer.output, W)
    connection = connections(d_to_i, i_to_h, h_to_o)
    b = bpnn(connection, layer, clock=slow_clock)
    net.add(b)

    net.add(i_to_h)
    net.add(h_to_o)
    net.add(d_to_i)
    net.add(neurons)
    return neurons, layer, connection


def backPropagate_addMonitors(neurons, layer, connections):
    output_monitor = SpikeMonitor(layer.output)
    hidden_monitor = SpikeMonitor(layer.hidden)
    input_monitor  = SpikeMonitor(layer.input)
    rate           = PopulationRateMonitor(layer.input)
    output_rate    = PopulationRateMonitor(layer.output)

    net.add(output_monitor)
    net.add(hidden_monitor)
    net.add(input_monitor)
    net.add(rate)
    net.add(output_rate)

    return input_monitor, hidden_monitor, output_monitor, rate, output_rate

neurons, layer, connections = backPropagate_setup(2, 5, 1)
input_monitor, hidden_monitor, output_monitor, rate, output_rate = backPropagate_addMonitors(neurons, layer, connection)



connections.hidden_to_output.compress()
#connections.hidden_to_output.W[:] = rand(connections.hidden_to_output.W.shape[0], connections.hidden_to_output.W.shape[1])
connections.delay = rand(connections.hidden_to_output.W.shape[0], connections.hidden_to_output.W.shape[1])*ms

connections.input_to_hidden.compress()
#connections.input_to_hidden.W[:] = rand(connections.input_to_hidden.W.shape[0], connections.input_to_hidden.W.shape[1])
connections.delay = rand(connections.input_to_hidden.W.shape[0], connections.input_to_hidden.W.shape[1])*ms
## i)   Feed-forward computation
## ii)  Backpropagation to the output layer
## iii) Backpropagation to the hidden layer
## iv)  Weight updates

net.run(run_time, report='text')
subplot(221)
plot(rate.times/ms,rate.smooth_rate(5*ms)/Hz)

subplot(222)
raster_plot(output_monitor)
title('Output spikes')

subplot(223)
raster_plot(input_monitor)
title('Input spikes')

subplot(224)
plot(output_rate.times/ms,output_rate.smooth_rate(5*ms)/Hz)
title('Output Rate')

show()


