from neuron_layer import neuron_layer


class e_hidden(neuron_layer):
    def _forwardImplementation(self, inbuf, outbuf):
        print inbuf
        outbuf[:] = inbuf
    
    def _backwardImplementation(self, outerr, inerr, outbuf, inbuf):
        inerr[:] = outerr
