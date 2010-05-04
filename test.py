from optparse import OptionParser
import matplotlib.pyplot as plt
import spikeprop
import pstats
import cProfile
import numpy as np
import profile
import pylab


parser = OptionParser()
parser.add_option("-d", "--debug", dest="debug", action="store_true")
parser.add_option("-n", "--ng", dest="ng", action="store_true")
(options, args) = parser.parse_args()

def xor(which):
    if which == 0:
        input = np.array([0.0, 0.0, 0.0])
        desired = np.array([16.0])
    elif which == 1:
        input = np.array([0.0, 0.0, 6.0])
        desired = np.array([10.0])
    elif which == 2:
        input = np.array([0.0, 6.0, 0.0])
        desired = np.array([10.0])
    elif which == 3:
        input = np.array([0.0, 6.0, 6.0])
        desired = np.array([16.0])
    else:
        input = np.array([0.0, 6.0, 6.0])
        desired = np.array([16.0])

    return input, desired


t = 'from_python'

from spikeprop_ng import *
from spikeprop_modular_ import *

prop = spikeprop_faster(3, 5, 1)
prop.initialise_weights()
prop.clear_slopes()


def run_modular():
    ## so if  layer[0].prev = 3 which would be h
    ## then   layer[0].next = 5 which would be i
    ## and    layer[1].prev = 5 which would also be i
    ## lastly layer[1].next = 1 which would be j
    input  = layer(3, 5)
    output = layer(5, 1)
    prop = modular([input, output])
    iterations = 5000
    x = 0
    Total_error = 10
    while x < iterations and Total_error > 0.5 and prop.failed == False:
        for w in xrange(4):
            input, desired = spikeprop.xor(w)
            error = prop.backwards_pass(input, desired)
            print error
            if error == False:
                break

            Total_error += error
            
        print "XOR: %d Total Error: %fms" % (x, Total_error)
        x += 1
            

    
def run_test(threshold):
    iterations = 5000
    x = 0
    total_error = 10.0
    errors = []
    prop.threshold = threshold
    while x < iterations and total_error > 0.5 and prop.failed == False:
        total_error = 0.0
        error_per = 0.0
        for w in xrange(4):
            input, desired = spikeprop.xor(w)
            error = prop.adapt(input, desired) 
            if error == False:
                break
            error_per += error
            total_error += error
            
        errors.append(error_per/4.0)
        error_per = 0

            
        print "XOR: %d Total Error: %fms" % (x, total_error)
        x += 1
            
    if prop.failed == True:
        print "!!! Failed !!!"
        return
    return errors


            
if options.debug:
    cProfile.run("run_test()","Profile.prof")
    s = pstats.Stats("Profile.prof")
    s.strip_dirs().sort_stats("time").print_stats()
else:
    #sets = []
    #for threshold in xrange(20,50):
    #    plt.plot(run_test(threshold))
    #    plt.show()
    #run_test(50)   

    run_modular()


