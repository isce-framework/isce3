from dataclasses import dataclass


@dataclass(frozen=True)
class CommonPaths:
    """
    Properties to paths common to all InSAR products. 
    
    Attributes:
    ------
    - RootPath (str): root path
    - IdentificationPath (str):  identification group path 
    """

    RootPath: str = "/science/LSAR"
    IdentificationPath: str = f"{RootPath}/identification"


@dataclass(frozen=True)
class RIFGGroupsPaths(CommonPaths):
    """
    RIFG Product Groups Paths
    
    Attributes:
    ------
    - ProductName (str): product name (RIFG)
    - MetadataPath (str):  metadata group path 
    - AttitudePath (str): attitude group path
    - GeolocationGridPath (str): geolocation group path
    - OrbitPath (str): orbit group path
    - ProcessingInformationPath (str): processingInformation group path
    - AlgorithmsPath (str): algorithms group path
    - InputsPath (str): inputs group path
    - ParametersPath (str): parameters group path
    - SwathsPath (str): swaths group path
    """

    ProductName: str = "RIFG"
    MetadataPath: str = f"{CommonPaths.RootPath}/{ProductName}/metadata"
    AttitudePath: str = f"{MetadataPath}/attitude"
    GeolocationGridPath: str = f"{MetadataPath}/geolocationGrid"
    OrbitPath: str = f"{MetadataPath}/orbit"
    ProcessingInformationPath: str = f"{MetadataPath}/processingInformation"
    AlgorithmsPath: str = f"{ProcessingInformationPath}/algorithms"
    InputsPath: str = f"{ProcessingInformationPath}/inputs"
    ParametersPath: str = f"{ProcessingInformationPath}/parameters"
    SwathsPath: str = f"{CommonPaths.RootPath}/{ProductName}/swaths"
