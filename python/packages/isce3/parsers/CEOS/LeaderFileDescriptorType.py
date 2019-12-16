###SAR Leaderfile Descriptor
def LeaderFileDescriptorType():
    '''
    SAR Leaderfile Descriptor Record.
    http://www.ga.gov.au/__data/assets/pdf_file/0019/11719/GA10287.pdf Pg:3-23.
    '''
    from .BasicTypes import (BlankType, StringType, IntegerType, BinaryType, MultiType)  
    from .FileDescriptorType import FileDescriptorType

    return MultiType( FileDescriptorType().mapping + 
                    [('NumberDatasetSummaryRecords', IntegerType(6)),
                     ('DatasetSummaryRecordLength', IntegerType(6)),
                     ('NumberMapProjectionRecords', IntegerType(6)),
                     ('MapProjectionRecordLength', IntegerType(6)),
                     ('NumberPlatformPositionRecords', IntegerType(6)),
                     ('PlatformPositionRecordLength', IntegerType(6)),
                     ('NumberAttitudeRecords', IntegerType(6)),
                     ('AttitudeRecordLength', IntegerType(6)),
                     ('NumberRadiometricRecords', IntegerType(6)),
                     ('RadiometricRecordLength', IntegerType(6)),
                     ('NumberRadiometricCompensationRecords', IntegerType(6)),
                     ('RadiometricCompensationRecordLength', IntegerType(6)),
                     ('NumberDataQualityRecords', IntegerType(6)),
                     ('DataQualityRecordLength', IntegerType(6)),
                     ('NumberHistogramRecords', IntegerType(6)),
                     ('HistogramRecordLength', IntegerType(6)),
                     ('NumberRangeSpectraRecords', IntegerType(6)),
                     ('RangeSpectraRecordLength', IntegerType(6)),
                     ('NumberDEMRecords', IntegerType(6)),
                     ('DEMRecordLength', IntegerType(6)),
                     ('NumberRadarParameterUpdateRecords', IntegerType(6)),
                     ('RadarParameterUpdateRecordLength', IntegerType(6)),
                     ('NumberAnnotationRecords', IntegerType(6)),
                     ('AnnotationRecordLength', IntegerType(6)),
                     ('NumberDetProcessingRecords', IntegerType(6)),
                     ('DetProcessingRecordLength', IntegerType(6)),
                     ('NumberCalibrationRecords', IntegerType(6)),
                     ('CalibrationRecordLength', IntegerType(6)),
                     ('NumberGCPRecords', IntegerType(6)),
                     ('GCPRecordLength', IntegerType(6)),
                     ('blanks3', BlankType(60)),
                     ('NumberFacility1Records', IntegerType(6)),
                     ('Facility1RecordLength', IntegerType(8)), 
                     ('NumberFacility2Records', IntegerType(6)),
                     ('Facility2RecordLength', IntegerType(8)), 
                     ('NumberFacility3Records', IntegerType(6)),
                     ('Facility3RecordLength', IntegerType(8)),
                     ('NumberFacility4Records', IntegerType(6)),
                     ('Facility4RecordLength', IntegerType(8)), 
                     ('NumberFacility5Records', IntegerType(6)),
                     ('Facility5RecordLength', IntegerType(8)), 
                     ('NumberFacility6Records', IntegerType(6)),
                     ('Facility6RecordLength', IntegerType(8)), 
                     ('NumberFacility7Records', IntegerType(6)),
                     ('Facility7RecordLength', IntegerType(8)), 
                     ('NumberFacility8Records', IntegerType(6)),
                     ('Facility8RecordLength', IntegerType(8)), 
                     ('NumberFacility9Records', IntegerType(6)),
                     ('Facility9RecordLength', IntegerType(8)), 
                     ('NumberFacility10Records', IntegerType(6)),
                     ('Facility10RecordLength', IntegerType(8)),
                     ('blanks4', BlankType(160)) ])

