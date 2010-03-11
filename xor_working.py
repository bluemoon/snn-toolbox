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
C           = 281*pF
gL          = 30*nS
taum        = C/gL
EL          = -70.6*mV
VT          = -80.4*mV
DeltaT      = 2*mV
Vcut        = VT+5*DeltaT
weight      = 0.7*mV
neurons     = 50
learn_rate  = 7
dt          = 1*ms 
gmax        = 0.01
run_time    = 1*second
re          = 0*mV
A           = []
mytime      = []
errors      = [0]
xor_history = []

tau_pre     = 7*ms
tau_post    = tau_pre
dA_pre      = .01
dA_post     = -dA_pre*tau_pre/tau_post*2.5

# Pick an electrophysiological behaviour
tauw, a, b, Vr = 144*ms, 4*nS, 0.0805*nA, -70.6*mV # Regular spiking (as in the paper)
##tauw,a,b,Vr=20*ms,4*nS,0.5*nA,VT+5*mV # Bursting
##tauw,a,b,Vr=144*ms,2*C/(144*ms),0*nA,-70.6*mV # Fast spiking

eqs = """
dvm/dt=(gL*(EL-vm)+gL*DeltaT*exp((vm-VT)/DeltaT)+I-w)/C : volt
dw/dt=(a*(vm-EL)-w)/tauw : amp
I : amp
"""

x1 = [off, off, on, on]
x2 = [off, on, off, on]

xor_time_table = []

def nextspike():
    base = 0*ms
    while True:
        r = randrange(0, len(x1))
        yield ((x1[r]*ms+base), (x2[r]*ms+base))
        base += 16*ms

def xor_diff(x):
    bools  = []
    target = []
    t      = []
    for idx, y in enumerate(x):
        base = idx*float(slow_clock.dt)
        t.append(base)

        if y > 0:
            bools.append(True)
            target.append(base+0.0010)
        if y < 0:
            bools.append(True)
            target.append(base+0.0010)
        if y == 0:
            bools.append(False)
            target.append(base+0.0016)

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
        # it is the iterator for neuron i, and nextspiketime is the stored time of the next spike
        for idx, spike in enumerate(self.nextspiketime):
            if spike is not None and float(spike) <= float(t):
                if idx not in self.fired:
                    firing[idx] = 1
                    self.fired.append(idx)


        if len(self.fired) == self.neurons and (float(t) % 16):
            self.nextspiketime = self.spiketimeiter.next()
            self.fired = []

        return where(firing)[0]

class STDPUpdater(SpikeMonitor):
    def __init__(self,source,C,vars,code,namespace,delay=0*ms):
        super(STDPUpdater,self).__init__(source, record=False, delay=delay)
        self._code=code # update code
        self._namespace=namespace # code namespace
        self.C=C
        
    def propagate(self,spikes):
        if len(spikes):
            self._namespace['spikes']=spikes
            self._namespace['w']=self.C.W
            exec self._code in self._namespace

class bpnn(NetworkOperation):
    def __init__(self, C, layer, clock=clock):
        NetworkOperation.__init__(self, lambda:None, clock=clock)
        #stdp = ExponentialSTDP(C, tau_pre, tau_post, dA_pre, dA_post, wmax=gmax, update='mixed', clock=fast_clock)
        #pre_mon = SpikeMonitor(stdp.pre_group)
        eq = """ 
        dA_pre/dt  = -A_pre/tau_pre   : 1
        """

        self.C = C
        self.pre  = NeuronGroup(len(C.source), model=eq, clock=self.clock)
        self.post = NeuronGroup(len(C.target), model=eq, clock=self.clock)
        
        self.layer = layer
        ## Monitors
        self.d_mon   = SpikeMonitor(layer.data)
        self.h_mon   = SpikeMonitor(layer.hidden)
        self.o_mon   = SpikeMonitor(layer.output)
        
        self.i_mon   = SpikeMonitor(C.source)
        self.j_mon   = SpikeMonitor(C.target)
        
    def from_ms(self, num):
        return num*10000
    
    def __call__(self):
        ## Constants
        tau_ = 7
        n_ = 0.05

        ## This is a major hack... the reason for this is the data is stored
        ## with a "unit" ms/mv etc. and when converted to a float it happens
        ## to be a really tiny decimal and is not useful in the calculations
        ms_   = lambda x: x*10000


        ## Get the differences between spike times, 
        ## this will give us the target value we need
        t_d     = xor_diff(self.d_mon.spiketimes[0] - self.d_mon.spiketimes[1])
        bools   = t_d[0]
        t_j_d   = t_d[1]
        d_k     = t_d[2]

        ## Current time conversion
        time    =  int(floor((float(self.clock.t)*1000)/16))
        print "time:", time

        t_j_a = lambda i: out_mon.spiketimes[i][time]
        t_i_a = lambda i: hidden_mon.spiketimes[i][time]
        t_h_a = lambda i: data_mon.spiketimes[i][time]

        
        gamma_j = lambda j: gamma_j_(connections.hidden_to_output, j)
        gamma_i = lambda j: gamma_i_(connections.input_to_hidden,  j)
        dw_ij   = lambda i,j: -(n_*(epsilon(ms_(t_j_a(j))-ms_(t_i_a(i))) * gamma_j(j)))
        

        def epsilon(t):
            return (t/tau_)*exp(1-(t/tau_))

        def depsilon(t):
            return -(exp(1-t/tau_)*t)/tau_**2 + (exp(1-t/tau_)/tau_)
    
        def dy(dy, dt):
            ## Formula [2] SpikeProp, Bohte.
            ## y.i(t) = epsilon(t-t.i)
            return depsilon(ms_(dt)-ms_(dy))

        def gamma_j_(c, j):
            ## Formula [12] SpikeProp, Bohte.
            ## 
            ##    t[j][desired] - t[j][actual]
            ##   ---------------------------------------------------
            ##    sum w.ij(pd y.i(t[j][actual])/pd t[j][actual])
            ##  i subset j 
            sum = 0.0  
            ## desired - actual
            top = t_j_d[time] - t_j_a(j)
            ## convert to a usable unit
            top = ms_(top)
            ## from the last layer to this one
            for i in xrange(c.W.shape[0]):
                ## sum all the connection weights 
                ## times the derivative of epsilon
                sum += c.W[i,j] * dy(t_i_a(i), t_j_a(j)) 
                
            return top/sum

        def gamma_i_(connection, i):
            sum = 0.0
            for j in xrange(connections.hidden_to_output.W.shape[1]):
                sum += gamma_j(j) * connections.hidden_to_output.W[i,j] * depsilon(ms_(t_i_a(i))) 
                
            conn_sum = 0.0
            for h in xrange(connection.W.shape[0]):
                conn_sum += connection.W[h, i] * depsilon(ms_(t_i_a(i))) 

            return sum/conn_sum
        
        def error_():
            sum = 0.0
            for x in xrange(1):
                sum += ms_(t_j_a(x)) - ms_(t_j_d[time])
            return 0.5*(sum**2)

        if ms_(float(slow_clock.t)) > 32 and len(t_j_d) > time:
            for j in xrange(connections.hidden_to_output.W.shape[1]):
                print "Gamma_j:", gamma_j(j), "node:", j
                
            for j in xrange(connections.hidden_to_output.W.shape[1]):
                for i in xrange(connections.hidden_to_output.W.shape[0]):
                    print "delta w(%d,%d):" % (i,j), dw_ij(i, j)
                    #print optimize.fmin(dw,[0],args=(connection.hidden_to_output,i,j))
                    print ms_(t_j_a(j)),ms_(t_i_a(i))

            print "Error:", error_()
            #print "Gamma_i:", gamma_i(0)
            #print stdp.Ap
            print ms_(stdp.A_pre), ms_(stdp.A_post)#, ms_(stdp.tau_pre), ms_(stdp.tau_post)
            print ms_(t_j_d[time] - t_j_a(0))
            time += 1
        
class layers:
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
    neurons = NeuronGroup(input_neurons+hidden_neurons+output_neurons, model=eqs, threshold=Vcut, reset="vm=Vr;w+=b", freeze=True, clock=fast_clock)
    
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
    W = rand(len(layer.input),len(layer.hidden))*mV
    i_to_h = Connection(layer.input,  layer.hidden, structure='dense')
    i_to_h.connect(layer.input, layer.hidden, W)
    ## hidden to output
    W=rand(len(layer.hidden),len(layer.output))*mV
    h_to_o = Connection(layer.hidden, layer.output, structure='dense')
    h_to_o.connect(layer.hidden, layer.output, W)
    b = bpnn(h_to_o, layer, clock=slow_clock)
    net.add(b)
    #stdp = ExponentialSTDP(h_to_o, tau_pre, tau_post, dA_pre, dA_post, wmax=gmax, update='mixed', clock=fast_clock)
    #pre_mon  = SpikeMonitor(stdp.pre_group)
    #post_mon = SpikeMonitor(stdp.post_group)


    net.add(i_to_h)
    net.add(h_to_o)
    net.add(d_to_i)
    #net.add(stdp)
    #net.add(pre_mon)
    #net.add(post_mon)

    connection = connections(d_to_i, i_to_h, h_to_o)


    ## Lots of hacks till the end of bpnn_
    @network_operation(slow_clock)
    def bpnn_():
        #print pre_mon.spiketimes, post_mon.spiketimes

        ## Constants
        tau_ = 7
        n_ = 0.05

        ## This is a major hack... the reason for this is the data is stored
        ## with a "unit" ms/mv etc. and when converted to a float it happens
        ## to be a really tiny decimal and is not useful in the calculations
        ms_   = lambda x: x*10000


        ## Get the differences between spike times, 
        ## this will give us the target value we need
        t_d     = xor_diff(data_mon.spiketimes[0] - data_mon.spiketimes[1])
        bools   = t_d[0]
        t_j_d   = t_d[1]
        d_k     = t_d[2]

        ## Current time conversion
        time    =  int(floor((float(slow_clock.t)*1000)/16))
        print "time:", time

        t_j_a = lambda i: out_mon.spiketimes[i][time]
        t_i_a = lambda i: hidden_mon.spiketimes[i][time]
        t_h_a = lambda i: data_mon.spiketimes[i][time]

        
        gamma_j = lambda j: gamma_j_(connections.hidden_to_output, j)
        gamma_i = lambda j: gamma_i_(connections.input_to_hidden,  j)
        dw_ij   = lambda i,j: -(n_*(epsilon(ms_(t_j_a(j))-ms_(t_i_a(i))) * gamma_j(j)))
        

        def epsilon(t):
            return (t/tau_)*exp(1-(t/tau_))

        def depsilon(t):
            return -(exp(1-t/tau_)*t)/tau_**2 + (exp(1-t/tau_)/tau_)
    
        def dy(dy, dt):
            ## Formula [2] SpikeProp, Bohte.
            ## y.i(t) = epsilon(t-t.i)
            return depsilon(ms_(dt)-ms_(dy))

        def gamma_j_(c, j):
            ## Formula [12] SpikeProp, Bohte.
            ## 
            ##    t[j][desired] - t[j][actual]
            ##   ---------------------------------------------------
            ##    sum w.ij(pd y.i(t[j][actual])/pd t[j][actual])
            ##  i subset j 
            sum = 0.0  
            ## desired - actual
            top = t_j_d[time] - t_j_a(j)
            ## convert to a usable unit
            top = ms_(top)
            ## from the last layer to this one
            for i in xrange(c.W.shape[0]):
                ## sum all the connection weights 
                ## times the derivative of epsilon
                sum += c.W[i,j] * dy(t_i_a(i), t_j_a(j)) 
                
            return top/sum

        def gamma_i_(connection, i):
            sum = 0.0
            for j in xrange(connections.hidden_to_output.W.shape[1]):
                sum += gamma_j(j) * connections.hidden_to_output.W[i,j] * depsilon(ms_(t_i_a(i))) 
                
            conn_sum = 0.0
            for h in xrange(connection.W.shape[0]):
                conn_sum += connection.W[h, i] * depsilon(ms_(t_i_a(i))) 

            return sum/conn_sum
        
        def error_():
            sum = 0.0
            for x in xrange(1):
                sum += ms_(t_j_a(x)) - ms_(t_j_d[time])
            return 0.5*(sum**2)

        if ms_(float(slow_clock.t)) > 32 and len(t_j_d) > time:
            for j in xrange(connections.hidden_to_output.W.shape[1]):
                print "Gamma_j:", gamma_j(j), "node:", j
                
            for j in xrange(connections.hidden_to_output.W.shape[1]):
                for i in xrange(connections.hidden_to_output.W.shape[0]):
                    print "delta w(%d,%d):" % (i,j), dw_ij(i, j)
                    #print optimize.fmin(dw,[0],args=(connection.hidden_to_output,i,j))
                    print ms_(t_j_a(j)),ms_(t_i_a(i))

            print "Error:", error_()
            #print "Gamma_i:", gamma_i(0)
            #print stdp.Ap
            print ms_(stdp.A_pre), ms_(stdp.A_post)#, ms_(stdp.tau_pre), ms_(stdp.tau_post)
            print ms_(t_j_d[time] - t_j_a(0))
            time += 1
            
        
    #net.add(bpnn_)   
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

co = matrix(connections.hidden_to_output.W)
ci = matrix(connections.input_to_hidden.W)

def dsigmoid(y):
    return 1.0-y*y


connections.hidden_to_output.compress()
#connections.hidden_to_output.W[:] = rand(connections.hidden_to_output.W.shape[0], connections.hidden_to_output.W.shape[1])
#connections.delay = rand(connections.hidden_to_output.W.shape[0], connections.hidden_to_output.W.shape[1])

connections.input_to_hidden.compress()
#connections.input_to_hidden.W[:] = rand(connections.input_to_hidden.W.shape[0], connections.input_to_hidden.W.shape[1])
#connections.delay = rand(connections.input_to_hidden.W.shape[0], connections.input_to_hidden.W.shape[1])
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


#print correlogram(output_monitor, input_monitor, width=20*ms, bin=1*ms)

show()


