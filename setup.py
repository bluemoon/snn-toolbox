#from distutils.core import setup
#from distutils.extension import Extension
from setuptools import setup, Extension

from Cython.Distutils import build_ext

includes = [
    '/usr/lib/python2.6/site-packages/numpy/core/include/',
    '/usr/local/lib/python2.6/dist-packages/numpy/core/include/',
    'old/',
    ]
    
spikeprop_math = Extension("snn_toolbox.Math",
                         ["Math.pyx","spike_prop_.c"],
                         include_dirs=includes)

spikeprop = Extension("snn_toolbox.modular",
                      ["modular.pyx","spike_prop_.c"],
                      include_dirs=includes,
                      #libraries=['profiler'],
                      )
spikeprop_ng = Extension("snn_toolbox.ng",
                         ["ng.pyx","spike_prop_.c"],
                         include_dirs=includes)

base = Extension("snn_toolbox.base",
                         ["base.pyx"],
                         include_dirs=includes)

structure = Extension("snn_toolbox.spikeprop_types",
                         ["spikeprop_types.pyx"],
                         include_dirs=includes)



ext = [spikeprop, spikeprop_ng, spikeprop_math, structure, base]
    
setup(
    name = 'spikeprop',
    cmdclass = {'build_ext': build_ext},
    ext_modules = ext,
    packages=['snn_toolbox'],
    package_dir={'snn_toolbox': '.'},
    version='0.0.1e',
)
