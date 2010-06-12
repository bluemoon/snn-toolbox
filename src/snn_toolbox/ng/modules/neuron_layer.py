from module import module

class neuron_layer(module):
    def __init__(self, dim, name=None):
        module.__init__(self, dim, dim, name=name)
        
    
