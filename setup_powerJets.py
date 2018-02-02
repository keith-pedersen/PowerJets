# run this script via
# 		$ python3 setup_powerJets.py build_ext --inplace
# (or via command python2 if you want a python2 library).
# This creates a shared library pYqRand.so, which can be imported as a python module
# 		>>> impoprt pYqRand as pqr
#
# I have configured my Linux + GNU system to have the following directory structure
#    ~/local/include/package/xyz.hpp      <= softlinks to my library headers, segregated (so I #include "package/xyz.hpp")
#    ~/local/lib/xyz.so                   <= softlinks to my libraries (not segregrated, all in the same folder)
#    ~/local/pyLib/xyz.so                 <= softlinks to my Python libraries (also not segregated)
# Then I have to set a few environmental variables (in my .bash_profile)
#    LD_LIBRARY_PATH = $HOME/local/lib/
#    PYTHONPATH = $HOMES/local/pyLib
# This allows me to easily build against and link to the C++ libraries, 
# and to "import module" from any running Python shell

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

# Tihis build script leverages the pre-compiled pqRand library (libpqr.so), 
# a library which must be accessible from anywhere you intend to use pYqRand.
# This keeps the C-compilation of pYqRand.pyx to a minimal (so the Python module is smaller),
# and allows different compile flags to be used for the Cython build.s

setup(
  name = "powerJets",
  ext_modules=[
    Extension('powerJets',
              sources=['source/powerJets.pyx'],
              include_dirs = ['include/', '/home/keith/local/include/'],
              libraries = ['kdp', 'pJets', 'pqr', 'pythia8', 'QtCore', 'fastjet'],
              library_dirs = ['/home/keith/local/lib/'],
              extra_compile_args=['-g', '-std=c++11', '-msse4', '-mavx2', '-mfpmath=sse', '-mieee-fp', '-march=native', '-ftree-vectorize'], # -O2 is default
              language='c++')
       ],
  cmdclass = {'build_ext': build_ext}
)
