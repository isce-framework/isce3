#cython: language_level=3
#
# Author: Joshua Cohen
# Copyright 2017
#

from libcpp.string cimport string
from Serialization cimport *

def deserialize(pyIH5File h5file, isceobj, **kwargs):
    """
    High-level interface for deserializing generic ISCE objects from an HDF5
    Level-1 product.

    Args:
        h5file (pyIH5File):                     IH5File for product.
        isceobj:                                Any supported ISCE extension class.
        
    Return:
        None 
    """

    if isinstance(isceobj, pyEllipsoid):
        loadEllipsoid(h5file, isceobj)

    elif isinstance(isceobj, pyOrbit):
        loadOrbit(h5file, isceobj, **kwargs)

    elif isinstance(isceobj, pyPoly2d):
        loadPoly2d(h5file, isceobj, **kwargs)

    elif isinstance(isceobj, pyRadar):
        loadRadar(h5file, isceobj)

    elif isinstance(isceobj, pyImageMode):
        loadImageMode(h5file, isceobj, **kwargs)

    elif isinstance(isceobj, pyMetadata):
        loadMetadata(h5file, isceobj)

    elif isinstance(isceobj, pyIdentification):
        loadIdentification(h5file, isceobj)

    elif isinstance(isceobj, pyComplexImagery):
        loadComplexImagery(h5file, isceobj)

    else:
        raise NotImplementedError('No suitable deserialization method found.')

# --------------------------------------------------------------------------------
# Serialization functions for isce::core objects
# --------------------------------------------------------------------------------

def loadEllipsoid(pyIH5File h5file, pyEllipsoid ellps):
    """
    Load Ellipsoid parameters from HDF5 file.

    Args:
        h5file (pyIH5File):                     IH5File for product.
        ellps (pyEllipsoid):                    pyEllipsoid instance.

    Return:
        None
    """
    load(deref(h5file.c_ih5file), deref(ellps.c_ellipsoid))

def loadOrbit(pyIH5File h5file, pyOrbit orbit, orbit_type='POE',
              pyDateTime refEpoch=MIN_DATE_TIME):
    """
    Load Orbit parameters from HDF5 file.

    Args:
        h5file (pyIH5File):                     IH5File for product.
        orbit (pyOrbit):                        pyOrbit instance.
        orbit_type (Optional[str]):             Orbit type ('MOE', 'NOE', 'POE').
        refEpoch (Optional[pyDateTime]):        Reference epoch for computing UTC time.

    Return:
        None
    """
    load(deref(h5file.c_ih5file), deref(orbit.c_orbit), 
         <string> pyStringToBytes(orbit_type), deref(refEpoch.c_datetime))

def loadPoly2d(pyIH5File h5file, pyPoly2d poly, poly_name='skew_dcpolynomial'):
    """
    Load Poly2d parameters from HDF5 file.

    Args:
        h5file (pyIH5File):                     IH5File for product.
        poly (pyPoly2d):                        pyPoly2d instance.
        poly_name (str):                        H5 dataset name for polynomial.

    Return:
        None
    """
    load(deref(h5file.c_ih5file), deref(poly.c_poly2d),
         <string> pyStringToBytes(poly_name))

# --------------------------------------------------------------------------------
# Serialization functions for isce::radar objects
# --------------------------------------------------------------------------------

def loadRadar(pyIH5File h5file, pyRadar radar):
    """
    Load Radar parameters from HDF5 file.

    Args:
        h5file (pyIH5File):                     IH5File for product.
        radar (pyRadar):                        pyRadar instance.

    Return:
        None
    """
    load(deref(h5file.c_ih5file), deref(radar.c_radar))

# --------------------------------------------------------------------------------
# Serialization functions for isce::product objects
# --------------------------------------------------------------------------------

def loadImageMode(pyIH5File h5file, pyImageMode imageMode, mode='primary'):
    """
    Load ImageMode parameters from HDF5 file.

    Args:
        h5file (pyIH5File):                     IH5File for product.
        imageMode (pyImageMode):                pyImageMode instance.
        mode (Optional[str]):                   Mode from ('aux', 'primary')

    Return:
        None
    """
    load(deref(h5file.c_ih5file), deref(imageMode.c_imagemode),
         <string> pyStringToBytes(mode))

def loadMetadata(pyIH5File h5file, pyMetadata meta):
    """
    Load Metadata parameters from HDF5 file.
    
    Args:
        h5file (pyIH5File):                     IH5File for product.
        meta (pyMetadata):                      pyMetadata instance.

    Return:
        None
    """
    load(deref(h5file.c_ih5file), meta.c_metadata)

def loadIdentification(pyIH5File h5file, pyIdentification ID):
    """
    Load Identification data from HDF5 file.

    Args:
        h5file (pyIH5File):                     IH5File for product.
        ID (pyIdentification):                  pyIdentification instance.

    Return:
        None
    """
    load(deref(h5file.c_ih5file), ID.c_identification)

def loadComplexImagery(pyIH5File h5file, pyComplexImagery cpxImg):
    """
    Load ComplexImagery data from HDF5 file.

    Args:
        h5file (pyIH5File):                     IH5File for product.
        cpxImg (pyComplexImagery):              pyComplexImagery instance.

    Return:
        None
    """
    load(deref(h5file.c_ih5file), cpxImg.c_compleximagery)

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
