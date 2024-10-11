from __future__ import annotations
from collections.abc import Iterable
from isce3.core import DateTime

# Mapping between polarizations available in a product and the corresponding
# two-character abbreviation.
# NOTE Want to use set to avoid having to list all permutations, but dict keys
# must be immutable, so use frozenset.
POL_MAPPING = {
    frozenset({"HH"}): "SH",
    frozenset({"VV"}): "SV",
    frozenset({"HH", "HV"}): "DH",
    frozenset({"VV", "VH"}): "DV",
    frozenset({"LH", "LV"}): "CL",
    frozenset({"RH", "RV"}): "CR",
    frozenset({"HH", "HV", "VH", "VV"}): "QP",
    # GCOV may symmetrize cross-pol channel (FP).  Still call it QP.
    frozenset({"HH", "HV", "VV"}): "QP",
    # We'll also call the other 3-pol case QP, possible in custom processing.
    frozenset({"HH", "VH", "VV"}): "QP",
    # InSAR processes only co-pol channels for input QP.  InSAR ilename spec
    # calls this QD, although that term typically means AHH + BVV.
    frozenset({"HH", "VV"}): "QD",
    frozenset({}): "NA"
}


def get_polarization_code(polarizations: Iterable[str],
                          default: str = "XX") -> str:
    """
    Determine the two-character polarization code based on the polarization
    content of a product.

    Parameters
    ----------
    polarizations : iterable of str
        List of TxRx polarizations in a frequency subband, e.g., ["HH"]
    default : str
        Value to return if polarizations can't be matched to known combinations.

    Returns
    -------
    code : str
        Polarization code
    """
    return POL_MAPPING.get(frozenset(polarizations), default)


def format_datetime(time: DateTime) -> str:
    """
    Return the datetime string needed to populate a NISAR granule_id.

    Parameters
    ----------
    time : isce3.core.DateTime
        Time stamp

    Returns
    -------
    time_str : str
        Time stamp formatted "YYYYMMDDTHHMMSS" as specified in NISAR file naming
        conventions JPL D-102255B, e.g., like ISO 8601 without punctuation.
    """
    # strip fraction of second and punctuation
    return time.isoformat().split('.')[0].replace('-', '').replace(':', '')
