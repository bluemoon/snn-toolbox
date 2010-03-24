import spikeprop
import pstats, cProfile

prop = spikeprop.spikeprop(3, 5, 1, 16)


def run_test():
    iterations = 3
    total_error = 0.0
    x = 0
    while x < iterations and prop.failed == False:
        for w in xrange(4):
            input, desired = spikeprop.xor(w)
            error = prop.adapt(input, desired)
            print "XOR: %d Error: %d" % (w, error)
        
        x += 1
        
    if not prop.failed:
        print "!!! Failed !!!"
    
    for w in xrange(4):
        input, desired = spikeprop.xor(w)
        error = prop.no_adapt(input, desired)
        #prop.print_times()
        total_error += error
    
    print "total_error: %d" % total_error

cProfile.runctx("run_test()", globals(), locals(), "Profile.prof")
s = pstats.Stats("Profile.prof")
s.strip_dirs().sort_stats("time").print_stats()
