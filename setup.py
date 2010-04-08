from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

ext_modules = [Extension("spikeprop", ["spikeProp.pyx","spike_prop_.c"],                                       
                         include_dirs=[
                             '/usr/lib/python2.6/site-packages/numpy/core/include/',
                             '/usr/local/lib/python2.6/dist-packages/numpy/core/include/',
                             ],
                         libraries=['profiler'],
                         )]
#ext_modules = [Extension("spikeutil", ["prop_util.pyx"])]

setup(
    name = 'spikeprop',
    cmdclass = {'build_ext': build_ext},
    ext_modules = ext_modules
)
