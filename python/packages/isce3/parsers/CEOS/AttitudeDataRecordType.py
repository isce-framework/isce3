###SAR Leaderfile Descriptor
def AttitudeDataRecordHeaderType():
    '''
    Attitude Data Record.
    http://www.ga.gov.au/__data/assets/pdf_file/0019/11719/GA10287.pdf Pg:3-42.
    '''
    from .BasicTypes import IntegerType, MultiType
    from .CEOSHeaderType import CEOSHeaderType

    return MultiType( CEOSHeaderType().mapping + 
                    [('NumberOfAttitudeDataPoints', IntegerType(4))])

def AttitudeDataRecordStateVectorType():
    '''
    YPR/ YPR rate section of the record.
    '''
    from .BasicTypes import (IntegerType,
                             FloatType,
                             MultiType)

    return MultiType([('DayOfYear', IntegerType(4)),
                     ('MillisecondsOfDay', IntegerType(8)),
                     ('PitchDataQualityFlag', IntegerType(4)),
                     ('RollDataQualityFlag', IntegerType(4)),
                     ('YawDataQualityFlag', IntegerType(4)),
                     ('PitchInDegrees', FloatType(14)),
                     ('RollInDegrees', FloatType(14)),
                     ('YawInDegrees', FloatType(14)),
                     ('PitchRateDataQualityFlag', IntegerType(4)),
                     ('RollRateDataQualityFlag', IntegerType(4)),
                     ('YawRateDataQualityFlag', IntegerType(4)),
                     ('PitchRateInDegreespers', FloatType(14)),
                     ('RollRateInDegreespers', FloatType(14)),
                     ('YawRateInDegreespers', FloatType(14))])

