from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

ext_modules = [Extension("spikeprop", ["spikeprop.pyx"])]
#ext_modules = [Extension("spikeutil", ["prop_util.pyx"])]

setup(
    name = 'spikeprop',
    cmdclass = {'build_ext': build_ext},
    ext_modules = ext_modules
)
