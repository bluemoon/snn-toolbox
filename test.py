import spikeprop
import pstats, cProfile
import numpy as np
import profile

profile.Profile.bias = 2.999750025e-06

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
    prop = spikeprop.spikeprop(3, 5, 1, 16, learning_rate=0.01, threshold=50)
    prop.init_2()
    def run_test():
        iterations = 5000
        total_error = 0.0
        x = 0
        while x < iterations and prop.failed == False:
            c_error = 0
            for w in xrange(4):
                input, desired = spikeprop.xor(w)
                error = prop.adapt(input, desired)
                if error == False:
                    break
                c_error += error

            print "XOR: %d Error: %fms" % (x, c_error)
            
            x += 1
            
        if prop.failed == False:
            print "!!! Failed !!!"
            return
        
        for w in xrange(4):
            input, desired = spikeprop.xor(w)
            error = prop.no_adapt(input, desired)
            total_error += error
            prop.print_times()
            
        print "total_error: %d" % total_error
        
    #cProfile.run("run_test()","Profile.prof")
    #s = pstats.Stats("Profile.prof")
    #s.strip_dirs().sort_stats("time").print_stats()
    run_test()
