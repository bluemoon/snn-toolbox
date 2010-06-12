from neuron_layer import neuron_layer

class linear_layer(neuron_layer):
    def __init__(self, *args, **kwargs):
        neuron_layer.__init__(self, *args, **kwargs)
        
    def _forwardImplementation(self, inbuf, outbuf):
        outbuf[:] = inbuf
    
    def _backwardImplementation(self, outerr, inerr, outbuf, inbuf):
        print outerr, inerr
        inerr[:] = outerr
