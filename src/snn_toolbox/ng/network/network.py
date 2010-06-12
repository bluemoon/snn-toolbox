from numpy import *
from snn_toolbox.ng.utils.params import parameter_container
from snn_toolbox.ng.modules.module import module

import logging
import scipy

class network:
    def __init__(self, name=None, **args):
        self.inmodules = []
        self.outmodules = []
        self.modules = []
        self.connections = {}
        self.conns = {}
        self.name = name
        self.current = None
        
    def __str__(self):
        sortedByName = lambda itr: sorted(itr, key=lambda i: i.name)
        
        params = {
            'name': self.name,
            'modules': self.modulesSorted,
            'connections': 
                sortedByName(combineLists(
                    [sortedByName(self.connections[m]) for m in self.modulesSorted])),
        }
        
        
        return '<network>' 


    def __getitem__(self, attribute):
        for a in self.modules:
            if a.name == attribute:
                return a
    
    def _iter(self, reverse=False):
        if reverse:
            for m in reversed(self.modules):
                yield m
                yield self.findConnection('out', m)
        else:
            for m in self.modulesSorted:
                yield m
                for c in self.connections[m]:
                    yield c

            
    def addInputModule(self, m):
        if m not in self.inmodules:
            self.inmodules.append(m)
        self.addModule(m)
        
    def addOutputModule(self, m):
        if m not in self.outmodules:
            self.outmodules.append(m)
        self.addModule(m)
            
    def addModule(self, m):
        """Add the given module to the network."""
        if m not in self.modules:
            self.modules.append(m)
        if not m in self.connections:
            self.connections[m] = []
        if m.paramdim > 0:
            m.owner = self

        self.sorted = False

    def findConnection(self, which, module):
        for key,value in self.conns.items():
            for subkey, subvalue in value.items():
                if subkey == which and subvalue == module:
                    return key
                
    def addConnection(self, c):
        if c not in self.conns:
            self.conns[c] = {'in':c.in_module,'out':c.out_module}

        c.in_module.connection_tail = c
        c.out_module.connection_head = c

        if not c.in_module in self.connections:
            self.connections[c.in_module] = []

        self.connections[c.in_module].append(c)
        if c.paramdim > 0:
            c.owner = self
            
        self.sorted = False

        
    def _topologicalSort(self):
        graph = {}
        for node in self.modules:
            if not graph.has_key(node):
                graph[node] = [0]

        for c in combineLists(self.connections.values()):
            graph[c.in_module].append(c.out_module)
            graph[c.out_module][0] += 1

        # Find all roots (nodes with zero incoming arcs).
        roots = [node for (node, nodeinfo) in graph.items() if nodeinfo[0] == 0]
        
        # Make sure the ordering on all runs is the same.
        roots.sort(key=lambda x: x.name)        
        
        # Repeatedly emit a root and remove it from the graph. Removing
        # a node may convert some of the node's direct children into roots.
        # Whenever that happens, we append the new roots to the list of
        # current roots.
        self.modulesSorted = []
        while len(roots) != 0:
            root = roots[0]
            roots = roots[1:]
            self.modulesSorted.append(root)
            for child in graph[root][1:]:
                graph[child][0] -= 1
                if graph[child][0] == 0:
                    roots.append(child)
            del graph[root]

        if graph:
            raise Exception('Missed')

    def forward(self):
        for self.current in self._iter():

            self.current.forward()
            #self.current._forwardImplementation(self.current.inputbuffer,
            #                                    self.current.outputbuffer)
        
    def backward(self):
        for m in reversed(self.modulesSorted):
            for c in self.connections[m]:
                c.backward()
            m.backward()
            
        
    def backActivate(self, outerr):
        self.modules[-1].outputerror = outerr
        self.backward()
        return self.modules[0].inputerror.copy()
    
    def activate(self, input, desired):
        self.modules[0].inputbuffer = input
        self.modules[-1].desiredbuffer = desired
        
        self.forward()
        return self.modules[-1].outputbuffer.copy()

    @property
    def derivs(self):
        tmp = [pc._derivs.flatten() for pc in self.conns.keys()]
        return concatenate(tmp)
    
    @property
    def weights(self):
        tmp = [pc.weights.flatten() for pc in self.conns.keys()]
        return concatenate(tmp)


    @property
    def deltas(self):
        tmp = [pc.deltas for pc in self.conns.keys()]
        return tmp
    
    def set_weights(self, weights):
        tmp = [pc for pc in self.conns.keys()]
        start = 0
        for pc in tmp:
            length = len(pc.weights.flatten())
            pc.weights = weights[start:start+length].reshape(pc.weights.shape)
            start += length
            
        
    def sortModules(self):
        if self.sorted:
            return
        
        # Sort the modules.
        self._topologicalSort()
        # Sort the connections by name.
        for m in self.modules:
            self.connections[m].sort(key=lambda x: x.name)
            
        #tmp = [p for p in self._iter()]
        #print tmp
        """
        # Create a single array with all parameters.
        tmp = [pc.params for pc in self._iter()]
        total_size = sum(scipy.size(i) for i in tmp)
        parameter_container.__init__(self, total_size)
        #print tmp, total_size
        
        if total_size > 0:
            self.params[:] = concatenate(tmp)
            self._setParameters(self.params)
        
            # Create a single array with all derivatives.
            tmp = [pc.derivs for pc in self._iter()]
            self.resetDerivatives()
            self.derivs[:] = scipy.concatenate(tmp)
            self._setDerivatives(self.derivs)
        
        # TODO: make this a property; indim and outdim are invalid before 
        # .sortModules is called!
        # Determine the input and output dimensions of the network.
        self.indim = sum(m.indim for m in self.inmodules)
        self.outdim = sum(m.outdim for m in self.outmodules)

        self.indim = 0
        for m in self.inmodules:
            self.indim += m.indim
        self.outdim = 0
        for m in self.outmodules:
            self.outdim += m.outdim

        # Initialize the network buffers.
        self.bufferlist = []
        module.__init__(self, self.indim, self.outdim, name=self.name)
        
        self.sorted = True
        """
        self.indim = 0
        for m in self.inmodules:
            self.indim += m.indim
        self.outdim = 0
        for m in self.outmodules:
            self.outdim += m.outdim
            
        #module.__init__(self, self.indim, self.outdim)
        
    def resetDerivatives(self):
        for k,v in self.conns.items():
            k.reset_derivative()

            
def combineLists(lsts):
    """ combine a list of lists into a single list """
    new = []
    for lst in lsts:
        for i in lst:
            new.append(i)
    return new
