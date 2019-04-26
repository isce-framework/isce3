include isce.def

TESTS = \
    raster \

all: test

test: raster

raster:
	${PYTHON} -m pytest ./raster.py
