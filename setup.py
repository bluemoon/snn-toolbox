#from distutils.core import setup
#from distutils.extension import Extension
import os, sys
import time, glob
from time import gmtime, strftime
from setuptools import setup, Extension, find_packages
from Cython.Distutils import build_ext
from setup_utilities import *

sdist = False
aux = 'misc/auxillary.c'

def find_data_files(dest_path_base, source_path):
    retval = []
    for path, dirs, files in os.walk(source_path):
        if not path.startswith(source_path):
            raise AssertionError()
        dest_path = path.replace(source_path, dest_path_base)
        source_files = [os.path.join(path, f) for f in files]
        retval.append((dest_path, source_files))
        if '.svn' in dirs:
            dirs.remove('.svn')
    return retval

def version(postfix):
     v = strftime("%Y%m%d.%H%M", gmtime())
     return v+postfix

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
        Extension('cy.math', ['cy/math.pyx', aux], include_dirs=includes),
        Extension('cy.ng', ['cy/ng.pyx', aux], include_dirs=includes),
        ]

if not sdist:
    print "Updating Cython code...."
    deps = DependencyTree()
    queue = compile_command_list(modules_ext, deps)
    execute_list_of_commands(queue)

def add(dictionary, k, v):
    if dictionary.has_key(k):
        dictionary[k].append(v)
    else:
        dictionary[k] = [v]
        
#data_dict = {}
#data_folders = [x for x in [f for pattern in ['*/*.pxd', '*/*.pyx'] for f in glob.glob(pattern) ]]
#data = [add(data_dict, os.path.dirname(x), os.path.split(x)[-1]) for x in data_folders]
setup(
    name = 'snn_toolbox',
    cmdclass = {'build_ext': setup_build_ext},
    ext_modules = modules_ext,
    packages=find_packages('src'),
    package_dir={'': 'src'},
    zip_safe=False,
    version=version('c'),
    test_suite = 'nose.collector',
    setup_requires = ['nose>=0.10.4'],
    entry_points = {
        'console_scripts': [
            'snn_src = snn_toolbox.ng.main:main',
            'snn_modular = snn_toolbox.ng.main:modular',
            'snn_new = snn_toolbox.new.main:main'
            ],
        #'hestia.ui' : ['hestia.ui=hestia.ui']
        }
)
