from isce3.parsers.CEOS.CalibrationDataRecordType import CalibrationDataRecordHeaderType


def CalibrationChirpReplicaType(samples, totalLength=6146):
    '''
    Calibration Data Record.
    http://www.ga.gov.au/__data/assets/pdf_file/0019/11719/GA10287.pdf Pg: 3-45.
    '''
    from isce3.parsers.CEOS.BasicTypes import (BinaryType,
                                               BlankType,
                                               IntegerType,
                                               MultiType)

    return MultiType([('ReceivePolarization', IntegerType(1)),
                      ('ChirpReplica', BinaryType('>i2', count=2*samples)),
                      ('blanks', BlankType(totalLength-4*samples-1))])
     

def CalibrationDataRecordTrailerType():
    '''
    This is ALOS-1 L0 data specific.
    '''
    from isce3.parsers.CEOS.BasicTypes import BlankType

    return BlankType(836)

