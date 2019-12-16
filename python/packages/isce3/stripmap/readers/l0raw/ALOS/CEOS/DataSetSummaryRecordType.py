def DatasetSummaryRecordType():
    '''
    Data Set Summary Record.
    http://www.ga.gov.au/__data/assets/pdf_file/0019/11719/GA10287.pdf Pg: 3-26.
    '''
    from isce3.parsers.CEOS.BasicTypes import (BlankType, IntegerType, FloatType, MultiType)
    from isce3.parsers.CEOS.DatasetSummaryRecordType import DatasetSummaryRecordCommonType

    #Common part of CEOS header
    inlist = DatasetSummaryRecordCommonType().mapping 

    #ALOS specific trailer
    inlist += [('CalibrationDataIndicator', IntegerType(4)),
               ('StartLineNumberCalibrationUpper', IntegerType(8)),
               ('StopLineNumberCalibrationUpper', IntegerType(8)),
               ('StartLineNumberCalibrationBottom', IntegerType(8)),
               ('StopLineNumberCalibrationBottom', IntegerType(8)),
               ('PRFSwitchingIndicator', IntegerType(4)),
               ('LineLocatorOfPRFSwitching', IntegerType(8)),
               ('blanks14', BlankType(16)),
               ('YawSteeringModeFlag', IntegerType(4)),
               ('ParameterTableNumber', IntegerType(4)),
               ('NominalOffNadirAngle', FloatType(16)),
               ('AntennaBeamNumber', IntegerType(4)),
               ('blanks15', BlankType(28)),
               ('blanks16', BlankType(120)),
               ('NumberOfAnnotationPoints', IntegerType(8)),
               ('blanks17', BlankType(2082))]

    return MultiType( inlist )

