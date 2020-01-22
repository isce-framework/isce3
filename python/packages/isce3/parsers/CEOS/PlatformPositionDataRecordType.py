###SAR Platform Position Data Record 
def PlatformPositionDataRecordHeaderType():
    '''
    Platform Position Data Record.
    http://www.ga.gov.au/__data/assets/pdf_file/0019/11719/GA10287.pdf Pg:3-39.
    '''
    from .BasicTypes import (BlankType, StringType, 
                            IntegerType, BinaryType, 
                            FloatType, MultiType)  
    from .CEOSHeaderType import CEOSHeaderType

    return MultiType( CEOSHeaderType().mapping + 
                    [('OrbitalElementsDesignator' , StringType(32)),
                     ('OrbitalElement1', FloatType(16)),
                     ('OrbitalElement2', FloatType(16)),
                     ('OrbitalElement3', FloatType(16)),
                     ('OrbitalElement4', FloatType(16)),
                     ('OrbitalElement5', FloatType(16)),
                     ('OrbitalElement6', FloatType(16)),
                     ('NumberOfDataPoints', IntegerType(4)),
                     ('YearOfDataPoint', IntegerType(4)),
                     ('MonthOfDataPoint', IntegerType(4)),
                     ('DayOfDataPoint', IntegerType(4)),
                     ('DayOfYear', IntegerType(4)),
                     ('SecondsOfDay', FloatType(22)),
                     ('TimeIntervalBetweenDataPointsInSec', FloatType(22)),
                     ('ReferenceCoordinateSystem', StringType(64)),
                     ('GreenwichMeanHourAngle', FloatType(22)),
                     ('AlongTrackPositionErrorInm', FloatType(16)),
                     ('AcrossTrackPositionErrorInm', FloatType(16)),
                     ('RadialPositionErrorInm', FloatType(16)),
                     ('AlongTrackVelocityErrorInmpers', FloatType(16)),
                     ('AcrossTrackVelocityErrorInmpers', FloatType(16)),
                     ('RadialVelocityErrorInmpers', FloatType(16))])

def PlatformPositionDataRecordStateVectorType():
    '''
    Position / velocity section of the record.
    '''
    from .BasicTypes import FloatType, MultiType
    return MultiType([('PositionXInm', FloatType(22)),
                      ('PositionYInm', FloatType(22)),
                      ('PositionZInm', FloatType(22)),
                      ('VelocityXInmpers', FloatType(22)),
                      ('VelocityYInmpers', FloatType(22)),
                      ('VelocityZInmpers', FloatType(22))])
