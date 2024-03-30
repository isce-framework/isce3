##This is the first cut to get the HDF5 translation working.
##Signal Record Iterator can be better designed
##The code currently only iterates over the records - modification/ manipulation belongs to customer

class ImageFile(object):
    '''
    Class for parsing ALOS-2 L1.1 CEOS Imagefile.
    '''

    def __init__(self, filename):
        '''
        Initialize object with leader filename.
        '''
        import os

        #Save file name
        self.name = filename 

        #Save file size in bytes
        self.size = os.stat(filename).st_size

        #Leader file always seems to consist of same set of records.
        #https://www.eorc.jaxa.jp/ALOS-2/en/doc/fdata/PALSAR-2_xx_Format_CEOS_E_f.pdf
        self.fid = open(self.name, 'rb')

        #Leader file descriptor
        self.description = self.parseFileDescriptor()

        #Line counter 
        self.counter = 0

    def close(self):
        '''
        Close file object.
        '''
        self.fid.close()

    def parseFileDescriptor(self):
        '''
        Parse SAR Leaderfile descriptor record.
        '''
        from isce3.parsers.CEOS.ImageFileDescriptorType import ImageFileDescriptorType

        #Description of record - seems to be common across missions
        record = ImageFileDescriptorType()
            
        #Read the record
        record.fromfile(self.fid)

        #Sensor specific validators
        #https://www.eorc.jaxa.jp/ALOS-2/en/doc/fdata/PALSAR-2_xx_Format_CEOS_E_f.pdf
        assert(record.RecordSequenceNumber == 1)
        assert(record.FirstRecordType == 50)
        assert(record.RecordTypeCode == 192)
        assert(record.SecondRecordSubType == 18)
        assert(record.ThirdRecordSubType == 18)
        assert(record.RecordLength == 720)
        assert(record.RecordSequenceType == 'FSEQ')
        assert(record.RecordCodeType == 'FTYP')
        assert(record.RecordFieldType == 'FLGT')
        assert(record.SampleDataLineNumberLocator == "13 4PB")
        assert(record.SARChannelNumberLocator == "49 2PB")
        assert(record.TimeOfSARDataLineLocator == "45 4PB")
        assert(record.LeftFillCountLocator == "21 4PB")
        assert(record.RightFillCountLocator == "29 4PB")
        assert(record.SARDataFormatTypeCode == "C*8")
        assert(record.NumberOfRightFillBitsWithinPixel == 0)
        assert(record.NumberOfBytesPerDataGroup % record.NumberOfSamplesPerDataGroup == 0)
        assert(record.NumberOfBytesOfSARDataPerRecord % (record.NumberOfBytesPerDataGroup//record.NumberOfSamplesPerDataGroup) == 0)

        #Check length of record
        assert(self.fid.tell() == record.RecordLength)

        #Return the validated record
        return record

    def readNextLine(self):
        '''
        Read the next line from file.
        '''
        from isce3.stripmap.readers.l1.ALOS2.CEOS.SignalDataRecordType import SignalDataRecordType

        #Create record type with information from description
        bytesperpixel = self.description.NumberOfBytesPerDataGroup // self.description.NumberOfSamplesPerDataGroup
        pixels = self.description.NumberOfBytesOfSARDataPerRecord // bytesperpixel
        record = SignalDataRecordType(pixels=pixels,
                                      bytesperpixel=bytesperpixel)


        #Read from file
        record.fromfile(self.fid)
        self.counter = self.counter + 1

        #Sensor specific validators
        assert(record.RecordSequenceNumber == (self.counter+1))
        assert(record.FirstRecordType == 50)
        assert(record.RecordTypeCode == 10)
        assert(record.SecondRecordSubType == 18)
        assert(record.ThirdRecordSubType == 20)
        assert(record.ActualCountOfLeftFillPixels == 0)
        assert(record.SARChannelCode == 0)
        assert(record.ScanIDForScanSAR == 0)
        assert(record.OnboardRangeCompressedFlag == 0)
        assert(record.PulseTypeDesignator == 0)

        return record


