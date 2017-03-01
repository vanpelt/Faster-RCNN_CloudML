# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import os
from os.path import join as pjoin
import numpy as np
import subprocess
from setuptools import setup, Extension
from setuptools import find_packages
from setuptools.command.build_ext import build_ext
from setuptools.command.install import install as _install

class MyInstall(_install):
    def command(self, command_list):
        print "Running command: %s" % command_list
        subprocess.check_call(command_list)

    def run(self):
        _install.run(self)
        
        try:
            #To prevent libdc1394 error
            #self.execute(self.command, (["ln", "-f", "/dev/null", "/dev/raw1394"]))
            #For tkinter from matplotlib.pyplot
            self.execute(self.command, [['apt-get', 'update']])
            self.execute(self.command, [['apt-get', '-y', 'install', 'python-tk']])
        except:
            print "You will likely need to install python-tk on your system"

def find_in_path(name, path):
    "Find a file in a search path"
    #adapted fom http://code.activestate.com/recipes/52224-find-a-file-given-a-search-path/
    for dir in path.split(os.pathsep):
        binpath = pjoin(dir, name)
        if os.path.exists(binpath):
            return os.path.abspath(binpath)
    return None

def locate_cuda():
    """Locate the CUDA environment on the system

    Returns a dict with keys 'home', 'nvcc', 'include', and 'lib64'
    and values giving the absolute path to each directory.

    Starts by looking for the CUDAHOME env variable. If not found, everything
    is based on finding 'nvcc' in the PATH.
    """

    # first check if the CUDAHOME env variable is in use
    if 'CUDAHOME' in os.environ:
        home = os.environ['CUDAHOME']
        nvcc = pjoin(home, 'bin', 'nvcc')
    else:
        # otherwise, search the PATH for NVCC
        default_path = pjoin(os.sep, 'usr', 'local', 'cuda', 'bin')
        nvcc = find_in_path('nvcc', os.environ['PATH'] + os.pathsep + default_path)
        if nvcc is None:
          return None;
        home = os.path.dirname(os.path.dirname(nvcc))

    cudaconfig = {'home':home, 'nvcc':nvcc,
                  'include': pjoin(home, 'include'),
                  'lib64': pjoin(home, 'lib64')}
    for k, v in cudaconfig.iteritems():
        if not os.path.exists(v):
            return None;

    return cudaconfig

def locate_tf():
    """Get the TENSORFLOW include"""
    try:
        import tensorflow as tf
        return tf.sysconfig.get_include()
    except ImportError:
        return "/usr/local/include"

CUDA = locate_cuda()

# Obtain the numpy include directory.  This logic works across numpy versions.
try:
    numpy_include = np.get_include()
except AttributeError:
    numpy_include = np.get_numpy_include()

def customize_compiler_for_nvcc(self):
    """inject deep into distutils to customize how the dispatch
    to gcc/nvcc works.

    If you subclass UnixCCompiler, it's not trivial to get your subclass
    injected in, and still have the right customizations (i.e.
    distutils.sysconfig.customize_compiler) run on it. So instead of going
    the OO route, I have this. Note, it's kindof like a wierd functional
    subclassing going on."""

    # tell the compiler it can processes .cu
    self.src_extensions.append('.cu')

    # save references to the default compiler_so and _comple methods
    default_compiler_so = self.compiler_so
    super = self._compile

    # now redefine the _compile method. This gets executed for each
    # object but distutils doesn't have the ability to change compilers
    # based on source extension: we add it.
    def _compile(obj, src, ext, cc_args, extra_postargs, pp_opts):
        print extra_postargs
        if os.path.splitext(src)[1] == '.cu':
            # use the cuda for .cu files
            self.set_executable('compiler_so', CUDA['nvcc'])
            # use only a subset of the extra_postargs, which are 1-1 translated
            # from the extra_compile_args in the Extension class
            postargs = extra_postargs['nvcc']
        else:
            postargs = extra_postargs['gcc']

        super(obj, src, ext, cc_args, postargs, pp_opts)
        # reset the default compiler_so, which we might have changed for cuda
        self.compiler_so = default_compiler_so

    # inject our redefined _compile method into the class
    self._compile = _compile


# run the customize_compiler
class custom_build_ext(build_ext):
    def build_extensions(self):
        customize_compiler_for_nvcc(self.compiler)
        build_ext.build_extensions(self)

ext_modules = [
    Extension(
        "utils.cython_bbox",
        ["utils/bbox.pyx"],
        extra_compile_args={'gcc': ["-Wno-cpp", "-Wno-unused-function"]},
	include_dirs = [numpy_include]
	),
    Extension(
        "utils.cython_nms",
        ["utils/nms.pyx"],
        extra_compile_args={'gcc': ["-Wno-cpp", "-Wno-unused-function"]},
        include_dirs = [numpy_include]
    ),
    Extension(
        "nms.cpu_nms",
        ["nms/cpu_nms.pyx"],
        extra_compile_args={'gcc': ["-Wno-cpp", "-Wno-unused-function"]},
        include_dirs = [numpy_include]
    )
]

if CUDA:
    ext_modules.append(
        Extension('nms.gpu_nms',
            ['nms/nms_kernel.cu', 'nms/gpu_nms.pyx'],
            library_dirs=[CUDA['lib64']],
            libraries=['cudart'],
            language='c++',
            runtime_library_dirs=[CUDA['lib64']],
            # this syntax is specific to this build system
            # we're only going to use certain compiler args with nvcc and not with gcc
            # the implementation of this trick is in customize_compiler() below
            extra_compile_args={'gcc': ["-Wno-unused-function"],
                                'nvcc': ['-arch=sm_35',
                                         '--ptxas-options=-v',
                                         '-c',
                                         '--compiler-options',
                                         "'-fPIC'"]},
            include_dirs = [numpy_include, CUDA['include']]
        )
    )
    ext_modules.append(
        Extension('roi_pooling_layer.roi_pooling',
            ['roi_pooling_layer/roi_pooling_op_gpu.cu',
            'roi_pooling_layer/roi_pooling_op.cc'],
            library_dirs=[CUDA['lib64']],
            libraries=['cudart'],
            language='c++',
            runtime_library_dirs=[CUDA['lib64']],
            include_dirs=[locate_tf()],
            extra_compile_args={
                'gcc': ["-std=c++11", "-D GOOGLE_CUDA=1"],
                'nvcc': [
                    "-std=c++11",
                    "-arch=sm_37",
                    "-D GOOGLE_CUDA=1",
                    "-Xcompiler",
                    "-fPIC",
                    "--expt-relaxed-constexpr"
                ]
            }
        )
    )
else:
    ext_modules.append(
        Extension('roi_pooling_layer.roi_pooling',
            ['roi_pooling_layer/roi_pooling_op.cc'],
            language='c++',
            include_dirs=[locate_tf()],
            extra_compile_args={
                'gcc': ["-std=c++11", "-D GOOGLE_CUDA=1"]
            }
        )
    )

VERSION="0.1.0"

setup(
    name='fast_rcnn',
    version=VERSION,
    ext_modules=ext_modules,
    setup_requires=['setuptools>=18.0','cython','numpy'],
    packages=find_packages(),
    install_requires=['cython', 'numpy', 'scipy', 'pillow', 'boto', 'easydict', 'pyyaml','matplotlib','opencv-python'],
    # inject our custom triggers
    cmdclass={'build_ext': custom_build_ext, 'install': MyInstall},
)
