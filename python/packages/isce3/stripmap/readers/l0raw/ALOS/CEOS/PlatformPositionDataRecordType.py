from isce3.parsers.CEOS.PlatformPositionDataRecordType import (PlatformPositionDataRecordHeaderType, PlatformPositionDataRecordStateVectorType)

def PlatformPositionDataRecordTrailerType():
    '''
    Returns ALOS specific trailer.
    '''

    from isce3.parsers.CEOS.BasicTypes import BlankType, IntegerType, MultiType

    return MultiType([('blanks1', BlankType(18)),
                      ('LeapSecondFlag', IntegerType(1)),
                      ('blanks2', BlankType(579))])
