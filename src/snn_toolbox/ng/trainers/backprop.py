from trainer import trainer
from numpy import *

class GradientDescent(object):
        
    def __init__(self):
        """ initialize algorithms with standard parameters (typical values given in parentheses)"""
        
        # --- BackProp parameters ---
        # learning rate (0.1-0.001, down to 1e-7 for RNNs)
        self.alpha = 0.1
        
        # alpha decay (0.999; 1.0 = disabled)
        self.alphadecay = 1.0
    
        # momentum parameters (0.1 or 0.9)
        self.momentum = 0.0
        self.momentumvector = None
 
        # --- RProp parameters ---
        self.rprop = False
        # maximum step width (1 - 20)
        self.deltamax = 5.0
        # minimum step width (0.01 - 1e-6)
        self.deltamin = 0.01
        # the remaining parameters do not normally need to be changed
        self.deltanull = 0.1
        self.etaplus = 1.2
        self.etaminus = 0.5
        self.lastgradient = None
        
    def init(self, values):
        """ call this to initialize data structures *after* algorithm to use
        has been selected
        
        :arg values: the list (or array) of parameters to perform gradient descent on
                       (will be copied, original not modified)
        """
        assert isinstance(values, ndarray)
        self.values = values.copy()
        if self.rprop:
            self.lastgradient = zeros(len(values), dtype='float64')
            self.rprop_theta = self.lastgradient + self.deltanull      
            self.momentumvector = None
        else:
            self.lastgradient = None
            self.momentumvector = zeros(len(values))
            
    def __call__(self, gradient, error=None):            
        """ calculates parameter change based on given gradient and returns updated parameters """
        # check if gradient has correct dimensionality, then make array """
        assert len(gradient) == len(self.values)
        gradient_arr = asarray(gradient)
        
        if self.rprop:
            rprop_theta = self.rprop_theta
            
            # update parameters 
            self.values += sign(gradient_arr) * rprop_theta 

            # update rprop meta parameters
            dirSwitch = self.lastgradient * gradient_arr
            rprop_theta[dirSwitch > 0] *= self.etaplus
            idx =  dirSwitch < 0
            rprop_theta[idx] *= self.etaminus
            gradient_arr[idx] = 0

            # upper and lower bound for both matrices
            rprop_theta = rprop_theta.clip(min=self.deltamin, max=self.deltamax)

            # save current gradients to compare with in next time step
            self.lastgradient = gradient_arr.copy()
            
            self.rprop_theta = rprop_theta
        
        else:
            # update momentum vector (momentum = 0 clears it)
            self.momentumvector *= self.momentum
        
            # update parameters (including momentum)
            self.momentumvector += self.alpha * gradient_arr
            self.alpha *= self.alphadecay
        
            # update parameters 
            self.values += self.momentumvector
            
        return self.values

    descent = __call__
    
class backprop(trainer):
    def __init__(self, module, dataset=None, learningrate=0.01, lrdecay=1.0,
                 momentum=0., verbose=False, batchlearning=False,
                 weightdecay=0.):
        
        trainer.__init__(self, module)
        self.verbose = verbose
        self.batchlearning = batchlearning
        self.weightdecay = weightdecay
        self.epoch = 0
        self.totalepochs = 0
        self.descent = GradientDescent()
        self.descent.alpha = learningrate
        self.descent.momentum = momentum
        self.descent.alphadecay = lrdecay
        self.descent.init(module.weights)
        

    def _deltas(self):
        pass
    
    def train(self, dataset):
        self.module.resetDerivatives()
        errors = 0        
        ponderation = 0.

        for seq in dataset._provideSequences():
            e, p = self._calcDerivs(seq)
            errors += e
            ponderation += p            
            #gradient = self.module.derivs - self.weightdecay * self.module.weights
            #new = self.descent(gradient, errors)
            #self.module.set_weights(new)
            #self.module.resetDerivatives()
            
        return errors, ponderation
    
    def train_until_convergence(self, dataset):
        while True:
            print self.train(dataset)
            
    def _calcDerivs(self, seq):
        for sample in seq:
            self.module.activate(sample[0], sample[1])
            
        error = 0
        ponderation = 0.
        for offset, sample in reversed(list(enumerate(seq))):
            # need to make a distinction here between datasets containing
            # importance, and others
            target = sample[1]
            outerr = target - self.module.activate(sample[0], sample[1])            
            if len(sample) > 2:
                importance = sample[2]
                error += 0.5 * dot(importance, outerr ** 2)
                ponderation += sum(importance)
                self.module.backActivate(outerr * importance)                
            else:
                error += 0.5 * sum(outerr ** 2)
                ponderation += len(target)
                # FIXME: the next line keeps arac from producing NaNs. I don't
                # know why that is, but somehow the __str__ method of the 
                # ndarray class fixes something,
                str(outerr)
                self.module.backActivate(outerr)
                
        return error, ponderation
