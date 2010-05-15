#from distutils.core import setup
#from distutils.extension import Extension
import os, sys
from setuptools import setup, Extension
from Cython.Distutils import build_ext
from setup_utilities import *

sdist = False
aux = 'misc/auxillary.c'

includes = [
    numpy.get_include(),
    '/usr/lib/python2.6/site-packages/numpy/core/include/',
    '/usr/local/lib/python2.6/dist-packages/numpy/core/include/',
    os.getcwd() + 'cy/',
    os.getcwd() + 'modular/',
    'cy/',
    ]


modules_ext =  [
        Extension('modular.base', ['modular/base.pyx'], include_dirs=includes),
        Extension('modular.spike_network', ['modular/spike_network.pyx'], include_dirs=includes),
        Extension('modular.spike_types', ['modular/spike_types.pyx', aux], include_dirs=includes),
        Extension('modular.dataset', ['modular/dataset.pyx', aux], include_dirs=includes),
        Extension('cy.modular', ['cy/modular.pyx', aux], include_dirs=includes),
        Extension('cy.math', ['cy/math.pyx', aux], include_dirs=includes),
        Extension('cy.ng', ['cy/ng.pyx', aux], include_dirs=includes),
        ]

if not sdist:
    print "Updating Cython code...."
    deps = DependencyTree()
    queue = compile_command_list(modules_ext, deps)
    execute_list_of_commands(queue)

#import nose
#nose.main()
    
setup(
    name = 'spikeprop',
    cmdclass = {'build_ext': sage_build_ext},
    #ext_package='snn_toolbox',
    ext_modules = modules_ext,
    packages=['snn_toolbox', 'snn_toolbox.modular', 'snn_toolbox.cy'],
    package_dir={'snn_toolbox': 'py'},
    zip_safe=False,
    version='0.0.1.9a',
    test_suite = 'nose.collector',
    setup_requires = ['nose>=0.10.4'],
)
