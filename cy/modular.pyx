# encoding: utf-8
# cython: profile=False
# cython: boundscheck=False
# cython: wraparound=False
# cython: infer_types=True
include "../misc/conf.pxi"

cimport modular.spike_types as types

cimport cy.math
import  cy.math

#np.import_array()
    

## Checklist
##  [ ]Excitation

cdef class modular(cy.math.Math):
    ## H (input), I (hidden) and J (output)
    ## 1) Firing time is x.j(excitation) ≥ v , progressing from the
    ##    hidden layer firing times, the firing times of
    ##    the output layer can be found, for a
    ##    given set of input firing times.
    ##
    ## 2) Calculate δj for all outputs according to (12)
    ##    For each subsequent layer I = J − {1...2}
    ##
    ## 3) Calculate δi for all neurons in I according to (17)
    ## 4) For output layer J , adapt w.ij by Δw.ij = −η * Y.i(t.j) * δj
    ##    For each subsequent layer I = J − {1...2}
    ## 6) Adapt w.hi by Δw.ij = −η * Y.h(t.i) * δi (18)
    ##
    ##   ∂E
    ## -------  = ε.ij^k(t-t.i-d.ij^k)δ.{i,j}
    ## ∂w.ij^k 
    ##
    cdef bint last
    def __init__(self, list layers):
        self.layers    = layers
        self.threshold = 50
        self.failed    = False
        self.layer     = None
        self.layer_idx = 0
        
        self.output_layer = <types.layer>self.layers[-1]
        self.input_layer  = <types.layer>self.layers[0]
        self.last         = False
        #self.propagating_type = 'descent'
        #self.propagating_routine = getattr(self, self.propagating_type + '_propagate')
        self.layer_length = len(self.layers)
        #self.propagating_routine = self.descent_propagate
        
    property fail:
        def __get__(self):
            return py.PyBool_FromLong(self.failed)
        
    cdef bint neg_weights(self):
        return NEG_WEIGHTS
    
    #property neg_weights:
    #    def __get__(self):
    #        return NEG_WEIGHTS
    #@cy.profile(False)
    cdef bint last_layer(self):
        cdef bint last_layer
        if self.layer_idx == (self.layer_length - 1):
            return True
        else:
            return False

    cdef bint first_layer(self):
        cdef bint first_layer
        if self.layer_idx == 0:
            first_layer = True
        else:
            first_layer = False
        return first_layer

    cdef inline bint hidden_layer(self):
        if not self.first_layer() and not self.last_layer():
            return True
        else:
            return False
            
    cpdef double quickprop_propagate(self, int i, int j, int k, double actual, double spike, int delay, double delta):
    #cpdef double quickprop_propagate(self, sPropagate prop):

        cdef next_step = 0.0
        mu = 2.0
        shrink = mu /(1.0 + mu)
        decay = 0.0001
        
        weight = self.layer.weights[i,j,k]
        double_prime = self.layer.derivative[i,j,k]
        prime = self.error_weight_derivative(actual, 
                                             spike, 
                                             delay, 
                                             delta)

        d =  self.layer.derive_delta[i,j,k]
        p =  double_prime
        s =  prime + (decay * weight)

        self.layer.derive_delta[i, j, k] = next_step
        self.layer.derivative[i, j, k]   = prime
        return next_step

    cdef inline double descent_propagate(self, int i, int j, int k, double actual, double spike, int delay, double delta):
        ## 1) ncalls: 225280 tottime: 0.360 per: 0.000001598
        ## 2) ncalls: 304640 tottime: 0.489 per: 0.000001605
        cdef int layer, m
        cdef double delta_weight
        if not self.last:
            layer = self.layer.next.size
            m = j
        else:
            layer = self.layer.prev.size
            m = i

        delta_weight = (-self.layer.learning_rate * c_e(actual - spike - delay) * delta)
        if m >= (layer - IPSP):
            #debug("IPSP Call: neuron: %d layer: %d layer size: %d" % (m, self.layer_idx, layer))
            #delta_weight = -self.change(actual, spike, delay, delta)
            delta_weight = -delta_weight

    
        return delta_weight

    cdef void _delta_j(self):
        cdef int j
        
        self.layer = self.layers[-1]
        self.layer_idx = self.layer_length-1
        for j in xrange(self.layer.next.size):
            self.layer.deltas[j] = self.delta_j(j)
            
    cdef void _delta_i(self):
        cdef int idx
        for idx from self.layer_length-1 >= idx > -1:
            self.layer_idx = idx
            self.layer = self.layers[self.layer_idx]
            if not self.last_layer():
                for i in xrange(self.layer.next.size):
                    self.layer.deltas[i] = self.delta_i(i)
                

    cpdef backwards_pass(self, np.ndarray input_times, np.ndarray desired_times):
        cdef:
            int idx, i, j, k
            int prev_size, next_size
            double *weights, *prev_data, *next_data, *deltas, *weight_data
            double *weight_ptr, new_weight, delta_weight
            np.ndarray prev_time, next_time, delta_np
            double spike_time, actual_time, delta
            types.layer layer
            np.npy_intp *strides
            
        self.forward_pass(input_times, desired_times)
        
        if self.failed:
            return False
        
        self._delta_j()
        self._delta_i()
        ## Go through every layer backwards
        ## doing the following steps:        
        for idx from self.layer_length-1 >= idx > -1:
            ##  1) figure out what layer we are on
            self.layer = self.layers[idx]
            self.layer_idx = idx
            self.last = self.last_layer()
            #layer = self.layer
            
            ##  2) calculate the delta j or delta i depending on the 
            ##     previous step, if this is the last layer we use
            ##     equation 12, if this is input -> hidden, or hidden to hidden
            ##     then we use equation 17

            ##  3) then we go through all of the next neuron set(j) 
            ##     get the time(actual_time), then go through all the 
            ##     neurons in the previous set(i)
            ##     and get the previously calculated delta
            prev = <types.neurons>self.layer.prev
            next = <types.neurons>self.layer.next

            prev_size = prev.size
            next_size = next.size
            
            #next_time = <np.ndarray>next.time
            #prev_time = <np.ndarray>prev.time
            #delta_np  = <np.ndarray>self.layer.deltas

            next_data = <double *>self.layer.next.time.data
            prev_data = <double *>self.layer.prev.time.data
            deltas    = <double *>self.layer.deltas.data
            weight_data = <double *>self.layer.weights.data
            strides = np.PyArray_STRIDES(self.layer.weights)
            #print strides[0], strides[1], strides[2]
            for j in xrange(next_size):
                #actual_time = (<double *>np.PyArray_GETPTR1(self.layer.next.time, j))[0]
                actual_time = next_data[j]
                #delta = (<double *>np.PyArray_GETPTR1(delta_np, j))[0]
                delta = deltas[j]
                for i in xrange(prev_size):
                    ## 4) from there we go through all the synapses(k)
                    #actual_time = (<double *>np.PyArray_GETPTR1(prev_time, i))[0]
                    spike_time = prev_data[i]
                    for k in xrange(SYNAPSES):
                        ## 5) we then get the spike time of the last layer(spike_time)
                        ##    get the last weight(old_weight) and if we are on the last
                        ##    layer
                        delay = k + 1
                        weight_ptr = <double *>(np.PyArray_BYTES(self.layer.weights) + (i * strides[0]) + (j * strides[1]) + (k * strides[2]))
                        #weight_ptr = <double *>np.PyArray_GETPTR3(self.layer.weights, i, j, k)
                        #old_weight = self.layer.weights[i, j, k]
                        delta_weight = self.descent_propagate(i, j, k, actual_time, spike_time, delay, delta)                                         
                        IF NEG_WEIGHTS:
                            #self.layer.weights[i, j, k] = new_weight
                            weight_ptr = weight_ptr[0] + delta_weight
                        ELSE:
                            if new_weight >= 0.0:
                                weight_ptr[0] = weight_ptr[0] + delta_weight
                                #self.layer.weights[i, j, k] = <double>new_weight
                            else:
                                weight_ptr[0] = 0.0
                                #self.layer.weights[i, j, k] = <double>0.0
                                
                                
            
        return self.error()
    

    cdef void forward_pass(self, np.ndarray input_times, np.ndarray desired_times):
        ## The first layer will be the furthest most left
        ## and the last layer will be the furthest most right
        ## 0 and -1 respectively
        cdef:
            double time = 0
            double ot = 0
            double total = 0
            double spike_time
            double weight
            int h, i, k, z, delay
            int idx, prev_size, next_size, prev_ipsp
            double *prev_time, *next_time
            np.ndarray next_array
            np.ndarray prev_array
            np.ndarray layer_weights
            
            double *weights
            types.neurons prev, next
            np.npy_intp *strides
            
        self.input_layer.prev.time             = input_times
        self.output_layer.next.desired_time    = desired_times
        
        for idx in xrange(self.layer_length):
            self.layer = self.layers[idx]
            self.layer_idx = idx
            self.last = self.last_layer()
            
            prev = self.layer.prev
            next = self.layer.next
            prev_size = self.layer.prev.size
            next_size = self.layer.next.size
            
            
            prev_time = <double *>self.layer.prev.time.data
            next_time = <double *>self.layer.next.time.data

            ## we now need to run from layer to layer
            ## first we must go through the next layer size(i)
            ## then we go through the previous layer(h)
            ## get the firing time of the previous layer(h)
            ## and figure out if the sum of all in our current time(time)
            ## passes the threshold value which is calculated with 
            ## spikeprop_math.linkout but because we are a subclass of it
            ## it can be accessed through self
            prev_ipsp = (prev_size - IPSP)
            strides = np.PyArray_STRIDES(self.layer.weights)
            bytes = np.PyArray_BYTES(self.layer.weights)
            
            for i in range(next_size):
                total = 0.0
                time  = 0.0
                while (total < self.threshold and time < MAX_TIME):
                    total = 0.0
                    for h in range(prev_size):
                        spike_time = prev_time[h]
                        ## when the time is past the spike time
                        if time >= spike_time:
                            weights = <double *>(bytes + (h * strides[0]) + (i * strides[1]))
                            #weights = <double *>np.PyArray_GETPTR2(layer_weights, h, i)                            
                            #weights = weight_data[strides[1]*i]
                            ot = 0.0
                            for k from 0 <= k < SYNAPSES:
                                delay = k+1
                                weight = weights[k]
                                ot += (weight * c_e(time - spike_time - delay))

                            if self.last:
                                if h >= prev_ipsp:
                                    total -= ot
                                else:
                                    total += ot
                            else:
                                total += ot

                    ## now set the next layers spike time to the current time
                    ## XXX: check to see if this can be optimized    

                    time += TIME_STEP

                next_time[i] = (time - TIME_STEP)
                #self.layer.next.time[i] = time - TIME_STEP
                
                
                if time >= 50.0:
                    self.failed = True
                    break

                        
    cpdef error(self):
        cdef types.layer last_layer = self.layers[-1]
        cdef double total = 0.0
        for j in range(last_layer.next.size):
            total += pow((last_layer.next.time[j] - last_layer.next.desired_time[j]), 2.0, None)
            
        return (total/2.0)


