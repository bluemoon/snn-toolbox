from optparse import OptionParser
import spikeprop
import pstats
import cProfile
import numpy as np
import profile

parser = OptionParser()
parser.add_option("-d", "--debug", dest="debug", action="store_true")
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


if t == 'from_cpp':
    prop = spikeprop.spikeProp(3, 5, 1)
    while x < iterations and prop.failed == False:
        for w in xrange(4):
            input, desired = xor(w)
            error = prop.adapt(input, desired)
            print error
        x+=1

if t == 'from_python':
    prop = spikeprop.spikeprop_base(3, 5, 1, 4, learning_rate=1.0, threshold=50)
    prop.init_1()
    #input, desired = spikeprop.xor(1)
    #prop.train(input, desired)
    #error = prop.backwards_pass(input, desired) 
    def run_test():
        iterations = 5000
        x = 0
        total_error = 10.0
        while x < iterations and total_error > 2.0 and prop.failed == False:
            total_error = 0.0
            for w in xrange(4):
                input, desired = spikeprop.xor(w)
                error = prop.adapt(input, desired) 
                if error == False:
                    break
                
                total_error += error
                print "XOR: %d-%d Error: %fms" % (x,w, error)
                
            print "XOR: %d Total Error: %fms" % (x, total_error)
            x += 1
            
        if prop.failed == True:
            print "!!! Failed !!!"
            return
        
        for w in xrange(4):
            input, desired = spikeprop.xor(w)
            error = prop.forward_pass(input, desired)
            total_error += error
            prop.print_times()
            
        print "total_error: %d" % total_error
        
    if options.debug:
        cProfile.run("run_test()","Profile.prof")
        s = pstats.Stats("Profile.prof")
        s.strip_dirs().sort_stats("time").print_stats()
    else:
        run_test()


