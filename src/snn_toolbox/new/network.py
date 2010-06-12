from graph import Graph
class network:
    def __init__(self):
        self.in_layers  = []
        self.out_layers = []
        self.layers     = []
        self.connections = {}
        self.graph = Graph()
        
    def add_input_layer(self, m):
        if m not in self.in_layers:
            self.in_layers.append(m)
        self.add_layer(m)
        
    def add_output_layer(self, m):
        if m not in self.out_layers:
            self.out_layers.append(m)
        self.add_layer(m)
        
    def add_layer(self, m):
        if m not in self.layers:
            self.layers.append(m)
            
        self.graph.add_edge(m.prev, m.next, edge_data=m)
        m.layers = self.layers
        self.sorted = False
        
    def forward(self, *args):
        for left, right, edge in self.graph.walk_edges():
            edge.forward()
            
    def backward(self, *args):
        for left, right, edge in self.graph.walk_edges_reverse():
            edge.backward()

    
    def activate(self, input, desired):
        self.layers[0].prev.time = input
        self.layers[-1].next.desired_time = desired
        
        self.forward()
        return self.layers[-1].next.time
            
    def back_activate(self):
        self.backward()
        
    def __iadd__(self, other):
        self.add_layer(other)
        return self
