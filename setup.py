#from distutils.core import setup
#from distutils.extension import Extension
import os
from setuptools import setup, Extension
from Cython.Distutils import build_ext

includes = [
    '/usr/lib/python2.6/site-packages/numpy/core/include/',
    '/usr/local/lib/python2.6/dist-packages/numpy/core/include/',
    'old/',
    os.getcwd(),
    ]
    

def build_directory(directory='modular/', prefix='snn_toolbox.modular.'):
    includes.extend([os.getcwd() +'/'+ directory])
    modular_files = os.listdir(directory)
    exclude = ['base.pyx']
    other_source = ['misc/auxillary.c']
    modular = [x for x in modular_files if os.path.splitext(x)[-1] == '.pyx' ]
    with_aux = [x for x in modular if x not in exclude]
    with_aux = zip(with_aux, other_source)
    without_aux = [x for x in modular if x in exclude]
    with_aux.extend(without_aux)
    modular_extensions = [Extension(prefix + os.path.splitext(x)[0], [directory + x ], include_dirs=includes) for x in modular]
    return modular_extensions


ext = []
ext.extend(build_directory(directory='old/', prefix='snn_toolbox.old.'))
ext.extend(build_directory(directory='modular/', prefix='snn_toolbox.modular.'))

setup(
    name = 'spikeprop',
    cmdclass = {'build_ext': build_ext},
    ext_modules = ext,
    packages=['snn_toolbox', 'snn_toolbox.modular'],
    package_dir={'snn_toolbox': 'python', 'snn_toolbox.modular':'modular'},
    version='0.0.1.6c',
    test_suite = 'nose.collector',
    setup_requires = ['nose>=0.10.4'],
)
