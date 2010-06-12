from numpy import *
from numpy.random import *

class parameter_container:
    stdParams = 1.
    mutationStd = 0.1
    owner = None
    hasDerivatives = False
    
    def __init__(self, paramdim = 0, **args):
        self.paramdim = paramdim
        if paramdim > 0:
            self._params = zeros(self.paramdim)
            from snn_toolbox.ng.modules.module import module
            from snn_toolbox.ng.connections.connections import connection
            from snn_toolbox.ng.network.network import network
            if isinstance(self, module) or isinstance(self, connection) or isinstance(self, network):
                self.hasDerivatives = True
            if self.hasDerivatives:
                self._derivs = zeros(self.paramdim)
            self.randomize()
                   
    @property
    def params(self):
        """ @rtype: an array of numbers. """
        return self._params
    
    def __len__(self):
        return self.paramdim
    
    def _setParameters(self, p, owner = None):
        """ :key p: an array of numbers """
        if isinstance(p, list):
            p = array(p)
        assert isinstance(p, ndarray)      

        if self.owner == self:
            # the object owns it parameter array, which means it cannot be set, 
            # only updated with new values.  
            self._params[:] = p
        elif self.owner != owner:
            raise Exception("Parameter ownership mismatch: cannot set to new array.")
        else:
            self._params = p
            self.paramdim = size(self.params)
        
    @property
    def derivs(self):
        """ :rtype: an array of numbers. """
        return self._derivs
    
    def _setDerivatives(self, d, owner = None):
        """ :key d: an array of numbers of self.paramdim """
        assert self.owner == owner
        assert size(d) == self.paramdim
        self._derivs = d
    
    def resetDerivatives(self):
        """ :note: this method only sets the values to zero, it does not initialize the array. """
        print self
        assert self.hasDerivatives
        self._derivs *= 0
    
    def randomize(self):
        self._params[:] = randn(self.paramdim)*self.stdParams
        if self.hasDerivatives:
            self.resetDerivatives()
            
    def mutate(self):
        self._params += randn(self.paramdim)*self.mutationStd
            
