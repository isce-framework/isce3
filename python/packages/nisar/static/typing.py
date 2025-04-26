from collections.abc import Mapping, Sequence
from enum import Enum
from typing import TypedDict, Union


RunConfigScalar = Union[str, bool, int, float, None]
RunConfigList = Sequence["RunConfigValue"]
RunConfigDict = Mapping[str, "RunConfigValue"]
RunConfigValue = Union[RunConfigScalar, RunConfigList, RunConfigDict]


class WaterMaskResampleAlgorithm(str, Enum):
    """ """

    NEAR = "near"
    MODE = "mode"

    def __str__(self) -> str:
        return self.value


class Geo2RdrParamDict(TypedDict, total=False):
    """ """

    threshold: float
    maxiter: int


class Rdr2GeoParamDict(TypedDict, total=False):
    """ """

    threshold: float
    numiter: int
    extraiter: int
