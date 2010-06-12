from neuron_layer import neuron_layer


class e_output(neuron_layer):
    def _forwardImplementation(self, inbuf, outbuf):
        outbuf[:] = inbuf
    
    def _backwardImplementation(self, outerr, inerr, outbuf, inbuf):
        inerr[:] = outerr
        print inerr
