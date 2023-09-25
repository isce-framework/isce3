from dataclasses import dataclass


@dataclass
class CommonPaths:
    """
    Properties to paths common to all InSAR products

    Attributes
    ----------
    ProductName : str
        Product name
    RootPath : str
        Root path
    IdentificationPath : str
        Identification group path
    MetadataPath : str
        Metadata group path
    AttitudePath : str
        Attitude group path
    OrbitPath : str
        Orbit group path
    ProcessingInformationPath : str
        ProcessingInformation group path
    AlgorithmsPath : str
        Algorithms group path
    InputsPath : str
        Inputs group path
    ParametersPath : str
        Parameters group path
    """
    ProductName: str = ""
    RootPath: str = "/science/LSAR"

    @property
    def IdentificationPath(self):
        return f"{self.RootPath}/identification"

    @property
    def MetadataPath(self):
        return f"{self.RootPath}/{self.ProductName}/metadata"

    @property
    def AttitudePath(self):
        return f"{self.MetadataPath}/attitude"

    @property
    def OrbitPath(self):
        return f"{self.MetadataPath}/orbit"

    @property
    def ProcessingInformationPath(self):
        return f"{self.MetadataPath}/processingInformation"

    @property
    def AlgorithmsPath(self):
        return f"{self.ProcessingInformationPath}/algorithms"

    @property
    def InputsPath(self):
        return f"{self.ProcessingInformationPath}/inputs"

    @property
    def ParametersPath(self):
        return  f"{self.ProcessingInformationPath}/parameters"

@dataclass
class L1GroupsPaths(CommonPaths):
    """
    Properties to paths common to all level 1 InSAR products.

    Attributes
    ----------
    GeolocationGridPath : str
        Geolocation group path
    SwathsPath : str
        Swaths group path
    """
    @property
    def GeolocationGridPath(self):
        return f"{self.MetadataPath}/geolocationGrid"

    @property
    def SwathsPath(self):
        return f"{self.RootPath}/{self.ProductName}/swaths"

@dataclass
class L2GroupsPaths(CommonPaths):
    """
    Properties to paths common to all level 2 InSAR products.

    Attributes
    ----------
    RadarGridPath : str
        Radar grid path
    GridsPath : str
        Grids group path
    """
    @property
    def RadarGridPath(self):
        return f"{self.MetadataPath}/radarGrid"

    @property
    def GridsPath(self):
        return f"{self.RootPath}/{self.ProductName}/grids"

@dataclass
class RIFGGroupsPaths(L1GroupsPaths):
    """
    RIFG Product Groups Paths

    Attributes
    ----------
    ProductName : str
        Product name (RIFG)
    """
    ProductName: str = "RIFG"

@dataclass
class RUNWGroupsPaths(L1GroupsPaths):
    """
    RUNW Product Groups Paths

    Attributes
    ----------
    ProductName : str
        Product name (RUNW)
    """
    ProductName: str = "RUNW"

@dataclass
class ROFFGroupsPaths(L1GroupsPaths):
    """
    ROFF Product Groups Paths

    Attributes
    ----------
    ProductName : str
        Product name (ROFF)
    """
    ProductName: str = "ROFF"

@dataclass
class GUNWGroupsPaths(L2GroupsPaths):
    """
    GUNW Product Groups Paths

    Attributes
    ----------
    ProductName : str
        Product name (GUNW)
    """
    ProductName: str = "GUNW"

@dataclass
class GOFFGroupsPaths(L2GroupsPaths):
    """
    GOFF Product Groups Paths

    Attributes
    ----------
    ProductName : str
        Product name (GOFF)
    """
    ProductName: str = "GOFF"
