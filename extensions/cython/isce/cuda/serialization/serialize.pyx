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

    elif isinstance(isceobj, pyPoly2d):
        loadPoly2d(group, isceobj, **kwargs)

    elif isinstance(isceobj, pyRadar):
        loadRadar(group, isceobj)

    elif isinstance(isceobj, pyImageMode):
        loadImageMode(group, isceobj, **kwargs)

    elif isinstance(isceobj, pyMetadata):
        loadMetadata(group, isceobj)

    elif isinstance(isceobj, pyIdentification):
        loadIdentification(group, isceobj)

    elif isinstance(isceobj, pyComplexImagery):
        loadComplexImagery(group, isceobj)

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

def loadOrbit(pyIGroup group, pyOrbit orbit, orbit_type='POE',
              pyDateTime refEpoch=MIN_DATE_TIME):
    """
    Load Orbit parameters from HDF5 file.

    Args:
        group (pyIGroup):                       IH5File for product.
        orbit (pyOrbit):                        pyOrbit instance.
        orbit_type (Optional[str]):             Orbit type ('MOE', 'NOE', 'POE').
        refEpoch (Optional[pyDateTime]):        Reference epoch for computing UTC time.

    Return:
        None
    """
    loadFromH5(group.c_igroup, deref(orbit.c_orbit), <string> pyStringToBytes(orbit_type),
               deref(refEpoch.c_datetime))

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

# --------------------------------------------------------------------------------
# Serialization functions for isce::radar objects
# --------------------------------------------------------------------------------

def loadRadar(pyIGroup group, pyRadar radar):
    """
    Load Radar parameters from HDF5 file.

    Args:
        group (pyIGroup):                       IH5File for product.
        radar (pyRadar):                        pyRadar instance.

    Return:
        None
    """
    loadFromH5(group.c_igroup, deref(radar.c_radar))

# --------------------------------------------------------------------------------
# Serialization functions for isce::product objects
# --------------------------------------------------------------------------------

def loadImageMode(pyIGroup group, pyImageMode imageMode, mode='primary'):
    """
    Load ImageMode parameters from HDF5 file.

    Args:
        group (pyIGroup):                       IH5File for product.
        imageMode (pyImageMode):                pyImageMode instance.
        mode (Optional[str]):                   Mode from ('aux', 'primary')

    Return:
        None
    """
    loadFromH5(group.c_igroup, deref(imageMode.c_imagemode), <string> pyStringToBytes(mode))

def loadMetadata(pyIGroup group, pyMetadata meta):
    """
    Load Metadata parameters from HDF5 file.
    
    Args:
        group (pyIGroup):                       IH5File for product.
        meta (pyMetadata):                      pyMetadata instance.

    Return:
        None
    """
    loadFromH5(group.c_igroup, meta.c_metadata)

def loadIdentification(pyIGroup group, pyIdentification ID):
    """
    Load Identification data from HDF5 file.

    Args:
        group (pyIGroup):                       IH5File for product.
        ID (pyIdentification):                  pyIdentification instance.

    Return:
        None
    """
    loadFromH5(group.c_igroup, ID.c_identification)

def loadComplexImagery(pyIGroup group, pyComplexImagery cpxImg):
    """
    Load ComplexImagery data from HDF5 file.

    Args:
        group (pyIGroup):                       IH5File for product.
        cpxImg (pyComplexImagery):              pyComplexImagery instance.

    Return:
        None
    """
    loadFromH5(group.c_igroup, cpxImg.c_compleximagery)

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
