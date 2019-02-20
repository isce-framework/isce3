#cython: language_level=3
#
# Author: Joshua Cohen
# Copyright 2017
#

from libcpp.string cimport string
from Serialization cimport *

def deserialize(pyIGroup group, isceobj, **kwargs):
    """
    High-level interface for deserializing generic ISCE objects from an HDF5
    Level-1 product.

    Args:
        group (pyIGroup):                       IH5File for product.
        isceobj:                                Any supported ISCE extension class.
        
    Return:
        None 
    """

    if isinstance(isceobj, pyEllipsoid):
        loadEllipsoid(group, isceobj)

    elif isinstance(isceobj, pyOrbit):
        loadOrbit(group, isceobj, **kwargs)

    elif isinstance(isceobj, pyEulerAngles):
        loadEulerAngles(group, isceobj)

    elif isinstance(isceobj, pyPoly2d):
        loadPoly2d(group, isceobj, **kwargs)

    elif isinstance(isceobj, pyLUT1d):
        loadLUT1d(group, isceobj, **kwargs)

    else:
        raise NotImplementedError('No suitable deserialization method found.')

# --------------------------------------------------------------------------------
# Serialization functions for isce::core objects
# --------------------------------------------------------------------------------

def loadEllipsoid(pyIGroup group, pyEllipsoid ellps):
    """
    Load Ellipsoid parameters from HDF5 file.

    Args:
        group (pyIGroup):                       IH5File for product.
        ellps (pyEllipsoid):                    pyEllipsoid instance.

    Return:
        None
    """
    loadFromH5(group.c_igroup, deref(ellps.c_ellipsoid))

def loadOrbit(pyIGroup group, pyOrbit orbit):
    """
    Load Orbit parameters from HDF5 file.

    Args:
        group (pyIGroup):                       IH5File for product.
        orbit (pyOrbit):                        pyOrbit instance.

    Return:
        None
    """
    loadFromH5(group.c_igroup, deref(orbit.c_orbit))

def loadEulerAngles(pyIGroup group, pyEulerAngles euler):
    """
    Load Euler angles attitude parameters from HDF5 file.

    Args:
        group (pyIGroup):                       IH5File for product.
        euelr (pyEulerAngles):                  pyEulerAngles instance.

    Return:
        None
    """
    loadFromH5(group.c_igroup, deref(euler.c_eulerangles))

def loadPoly2d(pyIGroup group, pyPoly2d poly, poly_name='skew_dcpolynomial'):
    """
    Load Poly2d parameters from HDF5 file.

    Args:
        group (pyIGroup):                       IH5File for product.
        poly (pyPoly2d):                        pyPoly2d instance.
        poly_name (str):                        H5 dataset name for polynomial.

    Return:
        None
    """
    loadFromH5(group.c_igroup, deref(poly.c_poly2d), <string> pyStringToBytes(poly_name))

def loadLUT1d(pyIGroup group, pyLUT1d lut, name_coords='r0', name_values='skewdc_values'):
    """
    Load LUT1d data from HDF5 file.

    Args:
        group (pyIGroup):                       IH5File for product.
        lut (pyLUT1d):                          pyLUT1d instance.
        name_coords (str):                      H5 dataset for LUT coordinates.
        name_values (str):                      H5 dataset for LUT values.

    Return:
        None
    """
    loadFromH5(group.c_igroup, deref(lut.c_lut), <string> pyStringToBytes(name_coords),
               <string> pyStringToBytes(name_values))

# --------------------------------------------------------------------------------
# Serialization functions for isce::geometry objects
# --------------------------------------------------------------------------------

def loadTopo(pyTopo topo, metadata):
    """
    Load Topo parameters from XML string.
    
    Args:
        topo (pyTopo):                          pyTopo instance.
        metadata (str):                         XML metadata string.

    Return:
        None
    """
    load_archive[Topo](<string> pyStringToBytes(metadata), 'Topo', topo.c_topo)

def loadGeo2rdr(pyGeo2rdr geo, metadata):
    """
    Load Geo2rdr parameters from XML string.
    
    Args:
        geo (pyGeo2rdr):                        pyTopo instance.
        metadata (str):                         XML metadata string.

    Return:
        None
    """
    load_archive[Geo2rdr](pyStringToBytes(metadata), 'Geo2rdr', geo.c_geo2rdr)
   

# end of file
