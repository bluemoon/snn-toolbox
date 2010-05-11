
class network:
    def __init__(self, layers):
        self.layers    = layers
        self.threshold = 50
        self.failed    = False
        self.layer     = None
        self.layer_idx = 0
        
        self.output_layer = self.layers[-1]
        self.input_layer  = self.layers[0]

        self.layer_length = len(self.layers)
        self.propagating_routine = self.descent_propagate

        
    def forward_pass(self):
        pass
    
    def backwards_pass(self):
        pass
