###SAR Imagefile Descriptor
def ImageFileDescriptorType():
    '''
    SAR Imagefile Descriptor Record.
    http://www.ga.gov.au/__data/assets/pdf_file/0019/11719/GA10287.pdf Pg:3-49.
    '''
    from .BasicTypes import (BlankType, StringType, IntegerType, BinaryType, MultiType)  
    from .FileDescriptorType import FileDescriptorType

    return MultiType( FileDescriptorType().mapping + 
                    [('NumberOfSARDataRecords', IntegerType(6)),
                     ('SARDataRecordLength', IntegerType(6)),
                     ('blanks3', BlankType(24)),
                     ('NumberOfBitsPerSample', IntegerType(4)),
                     ('NumberOfSamplesPerDataGroup', IntegerType(4)),
                     ('NumberOfBytesPerDataGroup', IntegerType(4)),
                     ('OrderOfSamples', StringType(4)),
                     ('NumberOfSARChannelsInTile', IntegerType(4)), 
                     ('NumberOfLinesPerDataSet', IntegerType(8)),
                     ('NumberOfLeftBorderPixelsPerLine', IntegerType(4)), 
                     ('NumberOfDataGroupsPerLinePerChannel', IntegerType(8)),
                     ('NumberOfRightBorderPixelsPerLine', IntegerType(4)),
                     ('NumberOfTopBorderLines', IntegerType(4)),
                     ('NumberOfBottomBorderLines', IntegerType(4)),
                     ('InterleavingIndicator', StringType(4)),
                     ('NumberOfPhysicalRecordsPerLine', IntegerType(2)),
                     ('NumberOfPhysicalRecordsPerMultiChannelLine', IntegerType(2)),
                     ('NumberOfBytesOfPrefixDataPerRecord', IntegerType(4)),
                     ('NumberOfBytesOfSARDataPerRecord', IntegerType(8)),
                     ('NumberOfBytesOfSuffixDataPerRecord', IntegerType(4)),
                     ('PrefixSuffixRepeatFlag', StringType(4)),
                     ('SampleDataLineNumberLocator', StringType(8)),
                     ('SARChannelNumberLocator', StringType(8)),
                     ('TimeOfSARDataLineLocator', StringType(8)),
                     ('LeftFillCountLocator', StringType(8)),
                     ('RightFillCountLocator', StringType(8)),
                     ('PadPixelsPresentIndicator', StringType(4)),
                     ('blanks4', BlankType(28)),
                     ('SARDataLineQualityCodeLocator', StringType(8)),
                     ('CalibrationInformationFieldLocator', StringType(8)),
                     ('GainValuesFieldLocator', StringType(8)),
                     ('BiasValuesFieldLocator', StringType(8)),
                     ('SARDataFormatType', StringType(28)),
                     ('SARDataFormatTypeCode', StringType(4)),
                     ('NumberOfLeftFillBitsWithinPixel', IntegerType(4)),
                     ('NumberOfRightFillBitsWithinPixel', IntegerType(4)),
                     ('MaximumDataRangeOfPixel', IntegerType(8)),
                     ('blanks5', BlankType(272))])

