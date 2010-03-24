from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

ext_modules = [Extension("spikeprop", ["spikeProp.pyx"],
                         include_dirs=[
                             '/usr/local/lib/python2.6/dist-packages/numpy/core/include/',
                             ]
                         )]
#ext_modules = [Extension("spikeutil", ["prop_util.pyx"])]

setup(
    name = 'spikeprop',
    cmdclass = {'build_ext': build_ext},
    ext_modules = ext_modules
)
