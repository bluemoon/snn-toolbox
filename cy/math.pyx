# cython: profile=False
# cython: boundscheck=False
# cython: wraparound=False
# cython: infer_types=True

include "../misc/conf.pxi"


cdef class Math:
    cdef double e(self, double time):
        return c_e(time)
    
    cdef int sign(self, int number):
        """
        >>> m = Math()
        >>> m.sign(135235)
        1
        >>> m.sign(-45645)
        -1
        
        """
        if number > 0:
            return 1
        elif number < 0:
            return -1
        else:
            return 0

    cdef double spike_response_derivative(self, double time):
        cdef double response = 0
        if time <= 0:
            return response
        else:
            return c_e(time) * ((1.0/time) - (1.0/DECAY))

    cdef double y(self, double time, double spike, double delay):
        return c_e(time - spike - delay)
    
    cdef inline double excitation(self, double *weights, double spike, double time):
        cdef double weight, output = 0.0
        cdef int k, i, delay
        ## if time >= (spike + delay)
        ## delay_max = SYNAPSES
        ## the delay is 1...16
        ## if time >= (spike + {1...16})
        ## so i need to find the minimum delay size and
        ## start the loop from there

        i = int(time-spike)
        for k from 0 <= k < i:
            delay = k+1
            weight = weights[k]
            output += (weight * c_e(time-spike-delay))

        return output
    
    ##def excitation(self, weights, spike, time):
        ## if time >= (spike + delay)
        ## delay_max = SYNAPSES
        ## the delay is 1...16
        ## if time >= (spike + {1...16})
        ## so i need to find the minimum delay size and
        ## start the loop from there
        #self.shared_result.value = 0.0
     ##   output = 0.0

        #i = int(time-spike)
        #for k in xrange(i):
        
        # #start = Time.time()
        # #processes = [self.pool.apply_async(e_sub, args=(i, time, spike, weights)) for i in range(SYNAPSES)]
        # #results   = [x.get() for x in processes]
        # #summ = sum(results)
        # #end  = Time.time()
        
        # #print "Threaded: ", end-start
        # #return summ


        # #start = Time.time()
        # output = 0
        # for k in xrange(SYNAPSES):
        #     #self.queue.put([k, weights, spike, time])
            
        #     delay = k+1
        #     weight = weights[k]
        #     output += (weight * self.e(time - spike - delay))

        #     #time - spike - delay

        # #end = Time.time()
        # #print "Non-Threaded: ", end-start
        
        # return output

    cdef double excitation_derivative(self, double *weights, double spike_time, double time):
        cdef double output = 0.0
        cdef double weight
        cdef int k
        if time >= spike_time:
            for k in xrange(SYNAPSES):
                delay  = k + 1
                ## will fire when current time 
                ## (timeT) >= time of spike + delay otherwise zero
                if time >= (spike_time + delay):
                    output += (weights[k] * self.spike_response_derivative((time - spike_time - delay)))
                    ## else no charge
                    
        return output

    cdef delta_j(self, int j):
        delta_j_top = self.layer.next.desired_time[j] - self.layer.next.time[j]
        return delta_j_top / self.delta_j_bottom(j)

    cdef delta_j_bottom(self, int j):
        #assert self.last_layer
        ot = 0.0
        for i in range(self.layer.prev.size):
            e_derivative =  self.excitation_derivative(<double *>np.PyArray_GETPTR2(self.layer.weights, i, j), 
                            self.layer.prev.time[i], 
                            self.layer.next.time[j])

            if i >= (self.layer.prev.size - IPSP):
                #debug("IPSP Call: neuron: %d layer: %d layer size: %d" % (i, self.layer_idx, self.layer.prev.size))
                ot -= e_derivative
            else:
                ot += e_derivative

        return ot
 
    cdef delta_i_top(self, int i):
        #assert not self.last_layer
        ## the top of equation 17 is from i to j
        ## so in our case it would be from the current layer
        ## (self.layer.prev to self.layer.next)
        
        ot     = 0.0
        actual = 0.0
        
        next_layer = self.layers[self.layer_idx+1]
        spike_time = next_layer.prev.time[i]
        for j in xrange(next_layer.next.size):
            actual_time = next_layer.next.time[j]
            delta = next_layer.deltas[j]
            excitation_d = self.excitation_derivative(<double *>np.PyArray_GETPTR2(next_layer.weights, i, j), spike_time, actual_time)
            if i >= (self.layer.next.size - IPSP):
                #debug("IPSP Call: neuron: %d layer: %d layer size: %d" % (i, self.layer_idx, self.layer.next.size))
                ot = -excitation_d
            else:
                ot = excitation_d
                
            actual += (delta * ot)

        return actual
    
    cdef delta_i_bottom(self, int i):
        ## the bottom of equation 17 is from h to i
        actual = 0.0
        actual_time = self.layer.next.time[i]
        for h in xrange(self.layer.prev.size):
            spike_time = self.layer.prev.time[h]
            ot = self.excitation_derivative(<double *>np.PyArray_GETPTR2(self.layer.weights, h, i), spike_time, actual_time)
            actual += ot
        
        if i >= (self.layer.next.size - IPSP):
            #debug("IPSP Call: neuron: %d layer: %d layer size: %d" % (i, self.layer_idx, self.layer.next.size))
            return -actual
        else:
            return actual

    cdef delta_i(self, int i):
        ## this equation works as follows
        ## we go from i to j in the top equation
        ## we go from h to i in the bottom equation
        ## so we need to go from the current layer
        ##   self.layer.prev as h to self.layer.next as i
        ## and then
        ##   self.layer.next as i to self.layers[self.layer_idx+1].prev as j
        ##
        ## so if  layer[0].prev = 3 which would be h
        ## then   layer[0].next = 5 which would be i
        ## and    layer[1].prev = 5 which would also be i
        ## lastly layer[1].next = 1 which would be j

        actual = self.delta_i_top(i)/self.delta_i_bottom(i)
        return actual
            

    cdef double change(self, double actual_time, double spike_time, double delay, double delta):
        return (-self.layer.learning_rate * c_e(actual_time - spike_time -delay) * delta)

    cdef double error_weight_derivative(self, double actual_time, double spike_time, double delay, double delta):
        return c_e(actual_time - spike_time - delay) * delta
    
