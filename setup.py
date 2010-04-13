from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

includes = [
    '/usr/lib/python2.6/site-packages/numpy/core/include/',
    '/usr/local/lib/python2.6/dist-packages/numpy/core/include/',
    ]

spikeprop = Extension("spikeprop",
                      ["spikeProp.pyx","spike_prop_.c"],
                      include_dirs=includes,
                      #libraries=['profiler'],
                      )
spikeprop_ng = Extension("spikeprop_ng",
                         ["spikeprop_ng.pyx","spike_prop_.c"],
                         include_dirs=includes)

ext = [spikeprop, spikeprop_ng]
    
setup(
    name = 'spikeprop',
    cmdclass = {'build_ext': build_ext},
    ext_modules = ext,
)
