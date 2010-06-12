from optparse import OptionParser
import pstats
import cProfile
import numpy as np
import profile

parser = OptionParser()
parser.add_option("-d", "--debug", dest="debug", action="store_true")
parser.add_option("-n", "--ng", dest="ng", action="store_true")
(options, args) = parser.parse_args()

from modular.dataset import dataset
input = np.array([[0.0, 0.1, 0.1],
                    [0.0, 0.1, 6.0],
                    [0.0, 6.0, 0.1],
                    [0.0, 6.0, 6.0]
                    ])
desired = np.array([[16.0], [10.0], [10.0], [16.0]])
data = [input, desired]
xor = dataset(data)
xor.autoscale()

def xor(which):
    if which == 0:
        input = np.array([0.0, 0.1, 0.1])
        desired = np.array([16.0])
    elif which == 1:
        input = np.array([0.0, 0.1, 6.0])
        desired = np.array([10.0])
    elif which == 2:
        input = np.array([0.0, 6.0, 0.1])
        desired = np.array([10.0])
    elif which == 3:
        input = np.array([0.0, 6.0, 6.0])
        desired = np.array([16.0])
    else:
        input = np.array([0.0, 6.0, 6.0])
        desired = np.array([16.0])

    return input, desired


t = 'from_python'

from cy.ng import *
from cy.modular import *
from modular.spike_network import *
from modular.spike_types   import * 


prop = spikeprop_faster(3, 5, 1)
prop.initialise_weights()
#prop.clear_slopes()


def run_modular():
    ## so if  layer[0].prev = 3 which would be h
    ## then   layer[0].next = 5 which would be i
    ## and    layer[1].prev = 5 which would also be i
    ## lastly layer[1].next = 1 which would be j
    i = neurons(3)
    h = neurons(5)
    o = neurons(1)

    input  = layer(i, h)
    output = layer(h, o)

    prop = modular([input, output])
    prop2 = modular2([input, output])
    x = 0
    iterations = 1000
    total_error = 10

    while x < iterations and total_error > 0.5 and prop.fail == False:
        total_error = 0
        for w in xrange(4):
            input, desired = xor(w)
            error = prop.backwards_pass(input, desired)
            total_error += error
            
        print "XOR: %d Total Error: %fms" % (x, total_error)
        x += 1

        
if options.debug:
    cProfile.run("run_modular()","Profile.prof")
    s = pstats.Stats("Profile.prof")
    s.strip_dirs().sort_stats("time").print_stats()
else:
    run_modular()


