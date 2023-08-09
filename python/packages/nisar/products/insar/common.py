import pathlib
from dataclasses import dataclass
import isce3
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
    - ProductVersion (str): product version, default is '1.0'
    """

    ProductSpecificationVersion: str
    ProductType: str
    ProductVersion: str

    @classmethod
    def RIFG(cls):
        return cls("0.1", "RIFG", "0.1")

    @classmethod
    def ROFF(cls):
        return cls("0.1", "ROFF", "0.1")

    @classmethod
    def RUNW(cls):
        return cls("0.1", "RUNW", "0.1")

    @classmethod
    def GOFF(cls):
        return cls("0.1", "GOFF", "0.1")

    @classmethod
    def GUNW(cls):
        return cls("0.1", "GUNW", "0.1")


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

    if not path_obj.exists():
        err_str = f"{path_str} does not exist"
        raise FileNotFoundError(err_str)

    if not path_obj.is_file():
        err_str = f"{path_str} is not a file"
        raise FileNotFoundError(err_str)

    return str(path_obj)



