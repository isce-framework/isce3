from dataclasses import dataclass


@dataclass
class CommonPaths:
    """
    Properties to paths common to all InSAR products. 
    
    Attributes:
    ------
    - ProductName (str): product name
    - MetadataPath (str):  metadata group path 
    - AttitudePath (str): attitude group path
    - OrbitPath (str): orbit group path
    - ProcessingInformationPath (str): processingInformation group path
    - AlgorithmsPath (str): algorithms group path
    - InputsPath (str): inputs group path
    - ParametersPath (str): parameters group path
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
    Properties to paths common to all level1 InSAR products. 
    
    Attributes:
    ------
    - GeolocationGridPath (str): geolocation group path
    - SwathsPath (str): swaths group path
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
    Properties to paths common to all level2 InSAR products. 
    
    Attributes:
    ------
    - RadarGridPath (str): radar grid path
    - GridsPath (str): grids group path
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
    
    Attributes:
    ------
    - ProductName (str): product name (RIFG)
    """
    
    ProductName: str = "RIFG"


@dataclass
class RUNWGroupsPaths(L1GroupsPaths):
    """
    RUNW Product Groups Paths
    
    Attributes:
    ------
    - ProductName (str): product name (RUNW)
    """
    
    ProductName: str = "RUNW"
    
    
@dataclass
class ROFFGroupsPaths(L1GroupsPaths):
    """
    ROFF Product Groups Paths
    
    Attributes:
    ------
    - ProductName (str): product name (ROFF)
    """
    
    ProductName: str = "ROFF"
    
    
@dataclass
class GUNWGroupsPaths(L2GroupsPaths):
    """
    GUNW Product Groups Paths
    
    Attributes:
    ------
    - ProductName (str): product name (GUNW)
    """
    
    ProductName: str = "GUNW"
    
    
@dataclass
class GOFFGroupsPaths(L2GroupsPaths):
    """
    GOFF Product Groups Paths
    
    Attributes:
    ------
    - ProductName (str): product name (GOFF)
    """
    
    ProductName: str = "GOFF"