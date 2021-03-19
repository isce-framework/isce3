from isce3.parsers.CEOS.AttitudeDataRecordType import (AttitudeDataRecordHeaderType,
    AttitudeDataRecordStateVectorType)

def AttitudeDataRecordTrailerType(inlen):
    '''
    Attitude Data Record Trailer.
    https://www.eorc.jaxa.jp/ALOS-2/en/doc/fdata/PALSAR-2_xx_Format_CEOS_E_f.pdf
    '''

    from isce3.parsers.CEOS.BasicTypes import BlankType

    return BlankType(inlen)

