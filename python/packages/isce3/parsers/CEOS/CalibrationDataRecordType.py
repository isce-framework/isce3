###SAR Calibration Data Record
def CalibrationDataRecordHeaderType():
    '''
    Attitude Data Record.
    http://www.ga.gov.au/__data/assets/pdf_file/0019/11719/GA10287.pdf Pg:3-42.
    '''
    from .BasicTypes import (IntegerType,
                             StringType,
                             MultiType)
    from .CEOSHeaderType import CEOSHeaderType

    return MultiType( CEOSHeaderType().mapping + 
                    [('CalDataSequenceNumber', IntegerType(4)),
                     ('NumberOfValidSamples', IntegerType(4)),
                     ('StartDateTimeOfChirpReplicaData', StringType(17)),
                     ('StopDateTimeOfChirpReplicaData', StringType(17)),
                     ('ATTSetting', IntegerType(4)),
                     ('ALCSetting', IntegerType(1)),
                     ('AGCorMGC', IntegerType(1)),
                     ('RangePulseLengthInusec', IntegerType(4)),
                     ('BandwidthInMHz', IntegerType(4)),
                     ('SamplingFrequencyinMHz', IntegerType(4)),
                     ('QuantizationBits', IntegerType(4)),
                     ('NumberOfChirpReplicaDataGroups', IntegerType(4)),
                     ('NumberOfChirpReplicaSampleData', IntegerType(4))
                     ])

