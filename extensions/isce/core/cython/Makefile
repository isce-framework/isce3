# Makefile to call the setup scripts for building a Cythonized Python module out of isce::core
# Note that the module will eventually be named isce.core, not iscecore (so importing an Ellipsoid,
# e.g. will be 'from isce.core import Ellipsoid' not 'from iscecore import Ellipsoid'
# Author: Joshua Cohen

.PHONY : clean recurse

# Build Python module using setup file and clean all intermediary files. Note that we need to clean first
# before trying to build since we want to force Cython to recompile the primary iscecore.pyx -> iscecore.cpp
# (which it won't do unless the dependent files are changed)
all : clean
	mkdir -p export; \
	python3 build_lib.py build_ext --inplace; \
	mv *.so export; \
	cp lib_init.py export/__init__.py; \
	rm -rf build; \
	rm -f iscecore.cpp

# Build Python module using setup file, clean intermediary files, and move everything up to
# the main export/ directory. See above about needing to run 'clean' first
recurse : clean
	python3 build_lib.py build_ext --inplace; \
	mkdir -p ../$(MAINBUILDDIR)/cython; \
	mv *.so ../$(MAINBUILDDIR)/cython; \
	cp lib_init.py ../$(MAINBUILDDIR)/cython/__init__.py
	rm -rf build; \
	rm -f iscecore.cpp

# Clean module, auto-gen'ed .cpp, and build/ dir
clean : 
	rm -rf export; \
	rm -rf build; \
	rm -f iscecore.cpp; \
	rm -f *.so

