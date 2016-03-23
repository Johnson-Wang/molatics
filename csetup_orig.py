from distutils.core import setup, Extension
import numpy
include_dirs_numpy = [numpy.get_include()]

extension = Extension('mddos._ac',
                      extra_compile_args=['-fopenmp'],
                      extra_link_args=['-lgomp'],
                      include_dirs=['c/ac_h'] + include_dirs_numpy,
                      sources=['c/_ac_orig.c',
                               'c/ac/ac_orig.c'])



setup(name='mdos',
      version='0.0.1',
      description='This is the dos-from-md module.',
      author='Wang Xinjiang',
      author_email='xwangan@ust.hk',
      ext_modules=[extension])
