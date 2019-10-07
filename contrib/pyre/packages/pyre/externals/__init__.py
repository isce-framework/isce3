# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# the marker of component factories
from .. import foundry


# a trait descriptor suitable for collecting package categories and instance specifications
def catalog(**kwds):
    """
    Build a trait descriptor suitable for building a database of available external packages
    for each package category
    """
    # get the trait descriptors
    from ..traits import properties
    # a catalog is a dictionary mapping package categories to list of packages
    return properties.catalog(schema=package(), **kwds)


def dependencies(**kwds):
    """
    Build a trait descriptor suitable for building a database of external package choices for
    each package category
    """
    # get the trait descriptors
    from ..traits import properties
    # {dependencies} is a dictionary mapping package categories to package instances
    return properties.dict(schema=package(), **kwds)


def requirements(**kwds):
    """
    Build a trait descriptor suitable for describing the list of package categories on which
    applications depend
    """
    # get the trait descriptors
    from ..traits import properties
    # {requirements} is a list of package category names
    return properties.list(schema=properties.str(), **kwds)


# convenience
from .Package import Package as package
from .Tool import Tool as tool
from .Library import Library as library


# the package abstractions
def blas():
    """
    The BLAS package manager
    """
    # grab the protocol
    from .BLAS import BLAS as blas
    # and generate a facility
    return blas()

def cython():
    """
    The Cython package manager
    """
    # grab the protocol
    from .Cython import Cython as cython
    # and generate a facility
    return cython()

def gcc():
    """
    The GCC package manager
    """
    # grab the protocol
    from .GCC import GCC as gcc
    # and generate a facility
    return gcc()

def gsl():
    """
    The GSL package manager
    """
    # grab the protocol
    from .GSL import GSL as gsl
    # and generate a facility
    return gsl()

def hdf5():
    """
    The HDF5 package manager
    """
    # grab the protocol
    from .HDF5 import HDF5 as hdf5
    # and generate a facility
    return hdf5()

def metis():
    """
    The metis package manager
    """
    # grab the protocol
    from .Metis import Metis as metis
    # and generate a facility
    return metis()

def mpi():
    """
    The MPI package manager
    """
    # grab the protocol
    from .MPI import MPI as mpi
    # and generate a facility
    return mpi()

def parmetis():
    """
    The parmetis package manager
    """
    # grab the protocol
    from .ParMetis import ParMetis as parmetis
    # and generate a facility
    return parmetis()

def petsc():
    """
    The PETSc package manager
    """
    # grab the protocol
    from .PETSc import PETSc as petsc
    # and generate a facility
    return petsc()

def postgres():
    """
    The Postgres package manager
    """
    # grab the protocol
    from .Postgres import Postgres as postgres
    # and generate a facility
    return postgres()

def python():
    """
    The Python package manager
    """
    # grab the protocol
    from .Python import Python as python
    # and generate a facility
    return python()

def vtk():
    """
    The VTK package manager
    """
    # grab the protocol
    from .VTK import VTK as vtk
    # and generate a facility
    return vtk()


# end of file
