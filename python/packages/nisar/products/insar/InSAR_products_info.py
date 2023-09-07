from dataclasses import dataclass

import isce3

ISCE3_VERSION = isce3.__version__


@dataclass
class InSARProductsInfo:
    """
    A data class describing the basic information of InSAR product
    including the product level, specification version, type, version,
    and geocoded or not

    Attributes
    ----------
    ProductSpecificationVersion : str
        Product specification version (default is '1.0')
    ProductType : str
        Product type, one of 'RIFG', 'ROFF', 'RUNW', 'GOFF', 'GUNW'
    ProductLevel : str
        Product level, one of 'L1' and 'L2'
    ProductVersion : str
        Product version (default is '0.1')
    isGeocoded : bool
        Geocoded product or not (True or False)
    """
    ProductSpecificationVersion: str
    ProductType: str
    ProductLevel: str
    ProductVersion: str
    isGeocoded: bool

    @classmethod
    def Base(cls):
        return cls("1.0", "", "", "", False)

    @classmethod
    def RIFG(cls):
        return cls("1.0", "RIFG", "L1", "0.1", False)

    @classmethod
    def ROFF(cls):
        return cls("1.0", "ROFF", "L1", "0.1", False)

    @classmethod
    def RUNW(cls):
        return cls("1.0", "RUNW", "L1", "0.1", False)

    @classmethod
    def GOFF(cls):
        return cls("1.0", "GOFF", "L2", "0.1", True)

    @classmethod
    def GUNW(cls):
        return cls("1.0", "GUNW", "L2", "0.1", True)
