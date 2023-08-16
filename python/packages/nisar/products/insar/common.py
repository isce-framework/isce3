import pathlib
from dataclasses import dataclass

import isce3
import journal

ISCE3_VERSION = isce3.__version__


@dataclass
class InSARProductsInfo:
    """
    A data class describing the basic information of InSAR product
    including the product level, specification version, type, and version

    Attributes:
    ------
    - ProductSpecificationVersion (str):  product specification version
    - ProductType (str): product type, one of 'RIFG', 'ROFF', 'RUNW', 'GOFF', 'GUNW'
    - ProductLevel (str): product level, one of 'L1' and 'L2'
    - ProductVersion (str): product version, default is '1.0'
    - isGeocoded (bool): geocoded product or not (True or False)
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


def get_validated_file_path(path_str: str):
    """
    Function to check validated path
    Function will account for optional path strings that maybe None.
    If None, then raise the FileNotFoundError

    Parameters:
    ------
    - path_str (str): file path
    Return
    - validate file path
    """
    if path_str is None:
        return None

    path_obj = pathlib.Path(path_str)

    err = journal.error('nisar.product.insar')
    if not path_obj.exists():
        err_str = f"{path_str} does not exist"
        err.log(err_str)
        raise FileNotFoundError(err_str)

    if not path_obj.is_file():
        err_str = f"{path_str} is not a file"
        err.log(err_str)
        raise FileNotFoundError(err_str)

    return str(path_obj)



