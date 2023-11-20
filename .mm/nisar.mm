# -*- Makefile -*-

# project meta-data
nisar.major := 3
nisar.minor := 0

# nisar consists of python packages
nisar.packages := nisar.pkg
# libraries
nisar.libraries :=
# python extensions
nisar.extensions :=
# there are files that get copied verbatim
nisar.verbatim := nisar.defaults nisar.schema
# and test suites
nisar.tests :=

# the nisar python package
nisar.pkg.stem := nisar
nisar.pkg.root := python/packages/nisar/
nisar.pkg.ext :=
nisar.pkg.meta :=
nisar.pkg.drivers :=

# the default workflow configuration files
nisar.defaults.root := share/nisar/defaults/
nisar.defaults.staging := $(builder.dest.pyc)/nisar/workflows/defaults/

# the workflow schema
nisar.schema.root := share/nisar/schemas/
nisar.schema.staging := $(builder.dest.pyc)/nisar/workflows/schemas/

# end of file
