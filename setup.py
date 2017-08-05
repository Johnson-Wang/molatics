from distutils.core import setup, Extension
import numpy
include_dirs_numpy = [numpy.get_include()]

extension = Extension('realmd._ac',
                      extra_compile_args=['-fopenmp'],
                      extra_link_args=['-lgomp'],
                      include_dirs=['c/ac_h'] + include_dirs_numpy,
                      sources=['c/_ac.c',
                               'c/ac/ac.c'])

extension_mdfc = Extension('mdfc._mdfc',
                      extra_compile_args=['-fopenmp'],
                      extra_link_args=['-lgomp',
                                       '-lopenblas'],
                      include_dirs=['c/mdfc_h'] + include_dirs_numpy,
                      sources=['c/_mdfc.c',
                               'c/mdfc/lapack_wrapper.c',
                               'c/mdfc/fc2.c',
                               'c/mdfc/fc3.c',
                               'c/mdfc/force_constants.c',
                               'c/mdfc/mathfunc.c'])

setup(name='molatics',
      version='0.3',
      description='This is the molecular + lattice dynamics module.',
      author='Wang Xinjiang',
      author_email='xwangan@ust.hk',
      packages=['realmd', 'realmd.mddos', 'mdfc', 'realmd.mdkappa'],
      scripts=['scripts/mdos',
               'scripts/mkappa',
               'scripts/fca',
               'scripts/mdfc'],
      ext_modules=[extension,
                   extension_mdfc])
