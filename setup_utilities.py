import numpy, time
import os, sys

from distutils.command.build_ext import build_ext
from distutils.dep_util import newer_group
from types import ListType, TupleType
from distutils import log
import distutils.sysconfig
import hashlib
import pickle

extra_compile_args = []
ext_modules = []

NO_WARN = True
if NO_WARN and distutils.sysconfig.get_config_var('CC').startswith("gcc"):
    extra_compile_args.append('-w')

DEVEL = False
if DEVEL:
    extra_compile_args.append('-ggdb')


CYTHON_INCLUDE_DIRS=['/usr/lib/python2.6/site-packages/Cython/Includes/']
SITE_PACKAGES = '/usr/lib/python2.6/site-packages/'

#for m in ext_modules:
#    m.libraries = ['csage'] + m.libraries + ['stdc++', 'ntl']
#    m.extra_compile_args += extra_compile_args
#    if os.environ.has_key('SAGE_DEBIAN'):
#        m.library_dirs += ['/usr/lib','/usr/lib/eclib','/usr/lib/singular','/usr/lib/R/lib','%s/lib' % SAGE_LOCAL]
#    else:
#        m.library_dirs += ['%s/lib' % SAGE_LOCAL]


def md5_file(file):
    file_reference = open(file, 'rb')
    data = file_reference.read()
    file_reference.close()
    return hashlib.md5(data).hexdigest()

def pickle_dict(dictionary):
    output = open('setup.pickle', 'wb')
    pickle.dump(dictionary, output)
    output.close()

def unpickle():
    output = open('setup.pickle', 'rb')
    p = pickle.load(output)
    output.close()
    return p

def execute_list_of_commands_in_serial(command_list):
    """
    INPUT:
        command_list -- a list of commands, each given as a pair
           of the form [command, argument].
        
    OUTPUT:
        the given list of commands are all executed in serial
    """    
    for f,v in command_list:
        r = f(v)
        if r != 0:
            print "Error running command, failed with status %s."%r
            sys.exit(1)

def run_command(cmd):
    """
    INPUT:
        cmd -- a string; a command to run
        
    OUTPUT:
        prints cmd to the console and then runs os.system  
    """    
    print cmd
    return os.system(cmd)

def apply_pair(p):
    """
    Given a pair p consisting of a function and a value, apply
    the function to the value.

    This exists solely because we can't pickle an anonymous function
    in execute_list_of_commands_in_parallel below.
    """
    return p[0](p[1])

def execute_list_of_commands_in_parallel(command_list, nthreads):
    """
    INPUT:
        command_list -- a list of pairs, consisting of a
             function to call and its argument
        nthreads -- integer; number of threads to use
        
    OUTPUT:
        Executes the given list of commands, possibly in parallel,
        using nthreads threads.  Terminates setup.py with an exit code of 1
        if an error occurs in any subcommand.

    WARNING: commands are run roughly in order, but of course successive
    commands may be run at the same time.
    """
    print "Execute %s commands (using %s threads)"%(len(command_list), min(len(command_list),nthreads))
    from multiprocessing import Pool
    import twisted.persisted.styles #doing this import will allow instancemethods to be pickable
    p = Pool(nthreads)
    for r in p.imap(apply_pair, command_list):
        if r:
            print "Parallel build failed with status %s."%r
            sys.exit(1)

def number_of_threads():
    """
    Try to determine the number of threads one can run at once on this
    system (e.g., the number of cores).  If successful return that
    number.  Otherwise return 0 to indicate failure.

    OUTPUT:
        int
    """
    if hasattr(os, "sysconf") and os.sysconf_names.has_key("SC_NPROCESSORS_ONLN"): # Linux and Unix
        n = os.sysconf("SC_NPROCESSORS_ONLN") 
        if isinstance(n, int) and n > 0:
            return n
    try:
        return int(os.popen2("sysctl -n hw.ncpu")[1].read().strip())
    except: 
        return 0
    
def execute_list_of_commands(command_list):
    """
    INPUT:
        command_list -- a list of strings or pairs
    OUTPUT:
        For each entry in command_list, we attempt to run the command.
        If it is a string, we call os.system. If it is a pair [f, v],
        we call f(v). On machines with more than 1 cpu the commands
        are run in parallel.
    """
    t = time.time()
    """
    if not os.environ.has_key('MAKE'):
        nthreads = 1
    else:
        MAKE = os.environ['MAKE']
        z = [w[2:] for w in MAKE.split() if w.startswith('-j')]
        if len(z) == 0:  # no command line option
            nthreads = 1
        else:
            # Determine number of threads from command line argument.
            # Also, use the OS to cap the number of threads, in case
            # user annoyingly makes a typo and asks to use 10000
            # threads at once.
            try:
                nthreads = int(z[0])
                n = 2*number_of_threads()
                if n:  # prevent dumb typos.
                    nthreads = min(nthreads, n)
            except ValueError:
                nthreads = 1
    """
    nthreads = 3
    # normalize the command_list to handle strings correctly
    command_list = [ [run_command, x] if isinstance(x, str) else x for x in command_list ]
                
    if nthreads > 1:
        # parallel version
        execute_list_of_commands_in_parallel(command_list, nthreads)
    else:
        # non-parallel version
        execute_list_of_commands_in_serial(command_list)
    print "Time to execute %s commands: %s seconds"%(len(command_list), time.time() - t)

########################################################################
##
## Parallel gcc execution
##
## This code is responsible for making distutils dispatch the calls to
## build_ext in parallel. Since distutils doesn't seem to do this by
## default, we create our own extension builder and override the
## appropriate methods.  Unfortunately, in distutils, the logic of
## deciding whether an extension needs to be recompiled and actually
## making the call to gcc to recompile the extension are in the same
## function. As a result, we can't just override one function and have
## everything magically work. Instead, we split this work between two
## functions. This works fine for our application, but it means that
## we can't use this modification to make the other parts of Sage that
## build with distutils call gcc in parallel.
##
########################################################################
        
class sage_build_ext(build_ext):
    def build_extensions(self):
        # First, sanity-check the 'extensions' list
        self.check_extensions_list(self.extensions)

        # We require MAKE to be set to decide how many cpus are
        # requested.

        #if not os.environ.has_key('MAKE'):
        #    ncpus = 1
        #else:
        #    MAKE = os.environ['MAKE']
        #    z = [w[2:] for w in MAKE.split() if w.startswith('-j')]
        #    if len(z) == 0:  # no command line option
        #        ncpus = 1
        #    else:
        #        # Determine number of cpus from command line argument.
                # Also, use the OS to cap the number of cpus, in case
                # user annoyingly makes a typo and asks to use 10000
                # cpus at once.
        #        try:
        #            ncpus = int(z[0])
        #            n = 2*number_of_threads()
        #            if n:  # prevent dumb typos.
        #                ncpus = min(ncpus, n)
        #        except ValueError:
        #            ncpus = 1
        ncpus = 2
        import time
        t = time.time()

        if ncpus > 1:

            # First, decide *which* extensions need rebuilt at
            # all.
            extensions_to_compile = []
            for ext in self.extensions:
                need_to_compile, p = self.prepare_extension(ext)
                if need_to_compile:
                    extensions_to_compile.append(p)

            # If there were any extensions that needed to be
            # rebuilt, dispatch them using pyprocessing.
            if extensions_to_compile:
               from multiprocessing import Pool
               import twisted.persisted.styles #doing this import will allow instancemethods to be pickable
               p = Pool(min(ncpus, len(extensions_to_compile)))
               for r in p.imap(self.build_extension, extensions_to_compile):
                   pass

        else:
            for ext in self.extensions:
                need_to_compile, p = self.prepare_extension(ext)
                if need_to_compile:
                    self.build_extension(p)

        print "Total time spent compiling C/C++ extensions: ", time.time() - t, "seconds."

    def prepare_extension(self, ext):
        sources = ext.sources
        if sources is None or type(sources) not in (ListType, TupleType):
            raise DistutilsSetupError, \
                  ("in 'ext_modules' option (extension '%s'), " +
                   "'sources' must be present and must be " +
                   "a list of source filenames") % ext.name
        sources = list(sources)

        fullname = self.get_ext_fullname(ext.name)
        if self.inplace:
            # ignore build-lib -- put the compiled extension into
            # the source tree along with pure Python modules

            modpath = string.split(fullname, '.')
            package = string.join(modpath[0:-1], '.')
            base = modpath[-1]

            build_py = self.get_finalized_command('build_py')
            package_dir = build_py.get_package_dir(package)
            ext_filename = os.path.join(package_dir,
                                        self.get_ext_filename(base))
            relative_ext_filename = self.get_ext_filename(base)
        else:
            ext_filename = os.path.join(self.build_lib,
                                        self.get_ext_filename(fullname))
            relative_ext_filename = self.get_ext_filename(fullname)

        # while dispatching the calls to gcc in parallel, we sometimes
        # hit a race condition where two separate build_ext objects
        # try to create a given directory at the same time; whoever
        # loses the race then seems to throw an error, saying that
        # the directory already exists. so, instead of fighting to
        # fix the race condition, we simply make sure the entire
        # directory tree exists now, while we're processing the
        # extensions in serial.
        relative_ext_dir = os.path.split(relative_ext_filename)[0]
        prefixes = ['', self.build_lib, self.build_temp]
        for prefix in prefixes:
            path = os.path.join(prefix, relative_ext_dir)
            if not os.path.exists(path):
                os.makedirs(path)
                            
        depends = sources + ext.depends
        if not (self.force or newer_group(depends, ext_filename, 'newer')):
            log.debug("skipping '%s' extension (up-to-date)", ext.name)
            need_to_compile = False
        else:
            log.info("building '%s' extension", ext.name)
            need_to_compile = True

        return need_to_compile, (sources, ext, ext_filename)

    def build_extension(self, p):

        sources, ext, ext_filename = p

        # First, scan the sources for SWIG definition files (.i), run
        # SWIG on 'em to create .c files, and modify the sources list
        # accordingly.
        sources = self.swig_sources(sources, ext)

        # Next, compile the source code to object files.

        # XXX not honouring 'define_macros' or 'undef_macros' -- the
        # CCompiler API needs to change to accommodate this, and I
        # want to do one thing at a time!

        # Two possible sources for extra compiler arguments:
        #   - 'extra_compile_args' in Extension object
        #   - CFLAGS environment variable (not particularly
        #     elegant, but people seem to expect it and I
        #     guess it's useful)
        # The environment variable should take precedence, and
        # any sensible compiler will give precedence to later
        # command line args.  Hence we combine them in order:
        extra_args = ext.extra_compile_args or []

        macros = ext.define_macros[:]
        for undef in ext.undef_macros:
            macros.append((undef,))

        objects = self.compiler.compile(sources,
                                        output_dir=self.build_temp,
                                        macros=macros,
                                        include_dirs=ext.include_dirs,
                                        debug=self.debug,
                                        extra_postargs=extra_args,
                                        depends=ext.depends)

        # XXX -- this is a Vile HACK!
        #
        # The setup.py script for Python on Unix needs to be able to
        # get this list so it can perform all the clean up needed to
        # avoid keeping object files around when cleaning out a failed
        # build of an extension module.  Since Distutils does not
        # track dependencies, we have to get rid of intermediates to
        # ensure all the intermediates will be properly re-built.
        #
        self._built_objects = objects[:]

        # Now link the object files together into a "shared object" --
        # of course, first we have to figure out all the other things
        # that go into the mix.
        if ext.extra_objects:
            objects.extend(ext.extra_objects)
        extra_args = ext.extra_link_args or []

        # Detect target language, if not provided
        language = ext.language or self.compiler.detect_language(sources)

        self.compiler.link_shared_object(
            objects, ext_filename,
            libraries=self.get_libraries(ext),
            library_dirs=ext.library_dirs,
            runtime_library_dirs=ext.runtime_library_dirs,
            extra_postargs=extra_args,
            export_symbols=self.get_export_symbols(ext),
            debug=self.debug,
            build_temp=self.build_temp,
            target_lang=language)




#############################################
###### Dependency checking
#############################################

#CYTHON_INCLUDE_DIRS=[ SAGE_LOCAL + '/lib/python/site-packages/Cython/Includes/' ]

# matches any dependency
import re
dep_regex = re.compile(r'^ *(?:(?:cimport +([\w\. ,]+))|(?:from +([\w.]+) +cimport)|(?:include *[\'"]([^\'"]+)[\'"])|(?:cdef *extern *from *[\'"]([^\'"]+)[\'"]))', re.M)

class DependencyTree:
    """
    This class stores all the information about the dependencies of a set of 
    Cython files. It uses a lot of caching so information only needs to be
    looked up once per build. 
    """
    def __init__(self):
        self._last_parse = {}
        self._timestamps = {}
        self._deps = {}
        self._deps_all = {}
        self.root = os.path.abspath(os.getcwd()+'/../')

    def __getstate__(self):
        """
        Used for pickling. 
        
        Timestamps and deep dependencies may change between builds, 
        so we don't want to save those. 
        """
        state = dict(self.__dict__)
        state['_timestamps'] = {}
        state['_deps_all'] = {}
        return state

    def __setstate__(self, state):
        """
        Used for unpickling. 
        """
        self.__dict__.update(state)
        self._timestamps = {}
        self._deps_all = {}
        self.root = os.getcwd()
        print self.root

    def timestamp(self, filename):
        """
        Look up the last modified time of a file, with caching. 
        """
        if filename not in self._timestamps:
            try:
                self._timestamps[filename] = os.path.getmtime(filename)
            except OSError:
                self._timestamps[filename] = 0
        return self._timestamps[filename]

    def parse_deps(self, filename, verify=True):
        """
        Open a Cython file and extract all of its dependencies. 
        
        INPUT: 
            filename -- the file to parse
            verify   -- only return existing files (default True)
        
        OUTPUT:
            list of dependency files
        """
        # only parse cython files
        if filename[-4:] not in ('.pyx', '.pxd', '.pxi'):
            return []
        
        dirname = os.path.split(filename)[0]
        deps = set()

        try:
            dictionary = unpickle()
        except Exception, E:
            dictionary = {}
            
        if md5_file(filename) not in dictionary.values():
            dictionary[filename] = md5_file(filename)
            
        if filename.endswith('.pyx'):
            pxd_file = filename[:-4] + '.pxd'
            if os.path.exists(pxd_file):
                deps.add(pxd_file)
                    
        pickle_dict(dictionary)

        
        raw_deps = []
        f = open(filename)
        for m in dep_regex.finditer(open(filename).read()):
            groups = m.groups()
            modules = groups[0] or groups[1] # cimport or from ... cimport
            if modules is not None:
                for module in modules.split(','):
                    module = module.strip().split(' ')[0] # get rid of 'as' clause
                    if '.' in module and module != 'cython':
                        path = module.replace('.', '/') + '.pxd'                        
                        base_dependency_name = path
                    else:
                        if os.path.isdir("%s/%s" % (dirname, module)):
                            path = "%s/%s/" % (dirname, module)
                            base_dependency_name = "__init__.pxd"
                        else:
                            path = "%s/%s.pxd" % (dirname, module)
                            base_dependency_name = "%s.pxd" % module
                        
                    raw_deps.append((path, base_dependency_name))
            else: # include or extern from
                extern_file = groups[2] or groups[3]
                path = '%s/%s'%(dirname, extern_file)
                if not os.path.exists(path):
                    path = extern_file
                raw_deps.append((path, extern_file))

        for path, base_dependency_name in raw_deps:
            # if we can find the file, add it to the dependencies.
            path = os.path.normpath(path)
            #print path, base_dependency_name
            
            if os.path.exists(path):
                deps.add(path)
            # we didn't find the file locally, so check the
            # Cython include path. 
            else:
                found_include = False
                for idir in CYTHON_INCLUDE_DIRS:
                    new_path = os.path.normpath(idir + base_dependency_name)
                    if os.path.exists(new_path):
                        deps.add(new_path)
                        found_include = True
                        break
                # so we really couldn't find the dependency -- raise
                # an exception.
                
                if not found_include:
                    if path[-2:] != '.h':  # there are implicit headers from distutils, etc
                        raise IOError, "could not find dependency %s included in %s."%(path, filename)
        f.close()
        return list(deps)

    def immediate_deps(self, filename):
        """
        Returns a list of files directly referenced by this file. 
        """
        if (filename not in self._deps
                or self.timestamp(filename) < self._last_parse[filename]):
            self._deps[filename] = self.parse_deps(filename)
            self._last_parse[filename] = self.timestamp(filename)
        return self._deps[filename]

    def all_deps(self, filename, path=None):
        """
        Returns all files directly or indirectly referenced by this file. 
        
        A recursive algorithm is used here to maximize caching, but it is
        still robust for circular cimports (via the path parameter). 
        """
        if filename not in self._deps_all:
            circular = False
            deps = set([filename])
            if path is None:
                path = set([filename])
            else:
                path.add(filename)
            for f in self.immediate_deps(filename):
                if f not in path:
                    deps.update(self.all_deps(f, path))
                else:
                    circular = True
            path.remove(filename)
            if circular:
                return deps # Don't cache, as this may be incomplete
            else:
                self._deps_all[filename] = deps
        return self._deps_all[filename]

    def newest_dep(self, filename):
        """
        Returns the most recently modified file that filename depends on, 
        along with its timestamp. 
        """
        nfile = filename
        ntime = self.timestamp(filename)
        for f in self.all_deps(filename):
            if self.timestamp(f) > ntime:
                nfile = f
                ntime = self.timestamp(f)
        return nfile, ntime


#############################################
###### Build code
#############################################

def process_filename(f, m):
    base, ext = os.path.splitext(f)
    if ext == '.pyx':
        if m.language == 'c++':
            return base + '.cpp'
        else:
            return base + '.c'
    else:
        return f

def compile_command(p):
    """
    Given a pair p = (f, m), with a .pyx file f which is a part the
    module m, call Cython on f

    INPUT:
         p -- a 2-tuple f, m

    copy the file to SITE_PACKAGES, and return a string
    which will call Cython on it.
    """
    f, m = p
    if f.endswith('.pyx'):
        # process cython file

        # find the right filename
        outfile = f[:-4]
        if m.language == 'c++':
            outfile += ".cpp"
        else:
            outfile += ".c"
            
        print outfile
        # call cython, abort if it failed
        cmd = "python `which cython`  -X boundscheck=True -p -I%s -o %s %s"%(os.getcwd(), outfile, f)
        r = run_command(cmd)
        if r:
            return r

        # if cython worked, copy the file to the build directory
        pyx_inst_file = '%s/%s'%(SITE_PACKAGES, f)
        retval = os.system('cp %s %s 2>/dev/null'%(f, '/tmp/'))
        # we could do this more elegantly -- load the files, use
        # os.path.exists to check that they exist, etc. ... but the
        # *vast* majority of the time, the copy just works. so this is
        # just specializing for the most common use case.
        if retval:
            dirname, filename = os.path.split(pyx_inst_file)
            if not os.path.exists(dirname):
                os.makedirs(dirname)
            retval = os.system('cp %s %s 2>/dev/null'%(f, pyx_inst_file))
            if retval:
                raise OSError, "cannot copy %s to %s"%(f,pyx_inst_file)
        print "%s --> %s"%(f, pyx_inst_file)
        
    elif f.endswith(('.c','.cc','.cpp')):
        # process C/C++ file
        cmd = "touch %s"%f
        r = run_command(cmd)
    
    return r

def compile_command_list(ext_modules, deps):
    """
    Computes a list of commands needed to compile and link the
    extension modules given in 'ext_modules'
    """
    queue_compile_high = []
    queue_compile_med = []
    queue_compile_low = []

    for m in ext_modules:
        new_sources = []
        for f in m.sources:
            if f.endswith('.pyx'):
                dep_file, dep_time = deps.newest_dep(f)
                dest_file = "%s/%s"%(SITE_PACKAGES, f)
                dest_time = deps.timestamp(dest_file)
                if dest_time < dep_time:
                    if dep_file == f:
                        print "Building modified file %s."%f
                        queue_compile_high.append([compile_command, (f,m)])
                    elif dep_file == (f[:-4] + '.pxd'):
                        print "Building %s because it depends on %s."%(f, dep_file)
                        queue_compile_med.append([compile_command, (f,m)])
                    else:
                        print "Building %s because it depends on %s."%(f, dep_file)
                        queue_compile_low.append([compile_command, (f,m)])
            new_sources.append(process_filename(f, m))
        m.sources = new_sources
    return queue_compile_high + queue_compile_med + queue_compile_low

