import numpy as np
import modular
import spikeprop_math
import spikeprop_ng
import unittest

SYNAPSES = 16
### Tests
## [x] sign
## [x] srfd
## [x] y
## [x] link_out
## [x] link_out_d
## [x] error_weight_derivative
## [x] _e12
## [x] _e12bottom
## [x] _e17top
## [x] _e17bottom
## [x] _e17
## [x] change
## [x] error


### Modular
## [x] first_layer
## [x] last_layer
class TestModular(unittest.TestCase):
    def setUp(self):
        self.math = modular.Math()
        
        i = modular.neurons(3)
        h = modular.neurons(5)
        o = modular.neurons(1)
        
        input  = modular.layer(i, h)
        output = modular.layer(h, o)
        
        self.modular = modular.modular([input, output])
        
    def test_learning_rate(self):
        for l in self.modular.layers:
            self.assertTrue(l.learning_rate > 0.0)
        
    def test_first_layer(self):
        self.modular.layer = self.modular.layers[0]
        self.modular.layer_idx = 0
        self.assertTrue(self.modular.first_layer)
        
        self.modular.layer = self.modular.layers[1]
        self.modular.layer_idx = 1
        self.assertTrue(self.modular.first_layer == False)
        
    def test_last_layer(self):
        self.modular.layer = self.modular.layers[0]
        self.modular.layer_idx = 0
        self.assertTrue(self.modular.last_layer == False)
        
        self.modular.layer = self.modular.layers[1]
        self.modular.layer_idx = 1
        self.assertTrue(self.modular.last_layer)

    def test_backwards_pass(self):
        input = np.array([1,2,3])
        desired = np.array([1])
        self.modular.backwards_pass(input, desired)
        first_layer  = self.modular.layers[0]
        second_layer = self.modular.layers[1]
        
        self.assertTrue(first_layer.next.time.all() == second_layer.prev.time.all())
        
class TestMaths(unittest.TestCase):
    def setUp(self):
        self.math = modular.Math()
        
    def test_sign_positive(self):
        self.assertTrue(self.math.sign(5) == 1)
        self.assertTrue(self.math.sign(934502394) == 1)
        self.assertTrue(self.math.sign(-934502394) != 1)

    def test_sign_negative(self):
        self.assertTrue(self.math.sign(-645) == -1)
        
    def test_sign_zero(self):
        self.assertTrue(self.math.sign(0) == 0)

class TestNewToOld(unittest.TestCase):
    def setUp(self):
        self.math = modular.Math()
        self.spike = spikeprop_ng.spikeprop_faster(3,5,1)
        
        i = modular.neurons(3)
        h = modular.neurons(5)
        o = modular.neurons(1)
        
        input  = modular.layer(i, h)
        output = modular.layer(h, o)

        self.modular = modular.modular([input, output])
        
    def setupTimes(self):
        time_desired = np.array([1])
        time_hidden  = np.array([3, 5, 7, 2, 1])
        time_output  = np.array([1.5])
        time_input   = np.array([2, 4, 6])

        
        self.spike.desired_time = time_desired
        self.spike.input_time   = time_input
        self.spike.hidden_time  = time_hidden
        self.spike.output_time  = time_output

        self.spike.hidden_weights = np.random.rand(5, 3, SYNAPSES)
        self.spike.output_weights = np.random.rand(1, 5, SYNAPSES)
        
        self.modular.layers[0].prev.time = time_input
        self.modular.layers[0].next.time = time_hidden
        self.modular.layers[1].next.time = time_output
        self.modular.layers[1].next.desired_time = time_desired

        self.modular.layers[0].weights = np.random.rand(3, 5, SYNAPSES)
        self.modular.layers[1].weights = np.random.rand(5, 1, SYNAPSES)
        self.modular.layer = self.modular.layers[1]
        
    def test_srfd(self):
        values = [3, 6, 12, 3543, 0, -1]
        for v in values:
            self.assertTrue(spikeprop_ng.srfd_(v) == self.math.spike_response_derivative(v))

    def test_y(self):
        values = [3, 6, 12, 3543, 0, -1]
        for v in values:
            self.assertTrue(spikeprop_ng.y_(v, v, v) == self.math.y(v, v, v))

    def test_link_out(self):
        spike_time  = 42.0
        actual_time = 45.0
        
        test_data = np.array([2.25657024,2.29774013,2.34550404,2.40091631,
                                2.46519961,2.53977173,2.46268021,2.31203762,
                                2.13301888,1.95480649,1.9885236,2.0,2.0,
                                2.0,2.0,2.0])
        
        left = self.math.excitation(test_data, spike_time, actual_time)
        right = self.spike.link_out(test_data, spike_time, actual_time)
        self.assertTrue(left == right)

    
    def test_link_out_d(self):
        spike_time  = 42.0
        actual_time = 45.0
        
        test_data = np.array([2.25657024,2.29774013,2.34550404,2.40091631,
                                2.46519961,2.53977173,2.46268021,2.31203762,
                                2.13301888,1.95480649,1.9885236,2.0,2.0,
                                2.0,2.0,2.0])
        left = self.math.excitation_derivative(test_data, spike_time, actual_time)
        right = self.spike.link_out_d(test_data, spike_time, actual_time)
        self.assertTrue(left == right)
    
    def test_error_weight_derivative(self):
        spike = 8.0
        time  = 9.0
        delay = 1.0
        delta = 6.0

        left = self.math.error_weight_derivative(time, spike, delay, delta)
        right = self.spike.error_weight_derivative(time, spike, delay, delta)
        self.assertTrue(left == right)
        
    def test_eq12(self):
        self.setupTimes()
        left  = self.spike._e12(0)
        right = self.modular.delta_j(0)

        self.assertTrue(left == right)
        
    def test_eq12_bottom(self):
        self.setupTimes()
        left  = self.spike._e12bottom(0)
        right = self.modular.delta_j_bottom(0)

        self.assertTrue(left == right)

    def test_eq17_bottom(self):
        self.setupTimes()
        left  = self.spike._e17bottom(0)
        right = self.modular.delta_i_bottom(0)

        self.assertTrue(left == right)
        
    def test_eq17_top(self):
        self.setupTimes()
        left  = self.spike._e17top(0, self.spike.delta_J)
        right = self.modular.delta_i_top(0)

        self.assertTrue(left == right)

    #def test_eq17(self):
    #    self.setupTimes()
    #    left  = self.spike._e17(0)
    #    right = self.modular.delta_i(0)
    #    print left, right
    #    self.assertTrue(left == right)

    def test_error(self):
        self.setupTimes()
        left  = self.spike.error()
        right = self.modular.error

        self.assertTrue(left == right)
        

    def test_change(self):
        actual = 7
        spike  = 8
        delay  = 9
        delta  = 6.7
        
        self.setupTimes()
        left  = self.spike.change(actual, spike, delay, delta)
        right = self.modular.change(actual, spike, delay, delta)

        self.assertTrue(left == right)
