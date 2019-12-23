from isce3.parsers.CEOS.AttitudeDataRecordType import (AttitudeDataRecordHeaderType,
    AttitudeDataRecordStateVectorType)

def AttitudeDataRecordTrailerType(inlen):
    '''
    Attitude Data Record Trailer.
    http://www.ga.gov.au/__data/assets/pdf_file/0019/11719/GA10287.pdf Pg: 3-42.
    '''

    from isce3.parsers.CEOS.BasicTypes import BlankType

    return BlankType(inlen)

