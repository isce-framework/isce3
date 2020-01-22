#!/usr/bin/env python3

def cmdLineParse():
    '''
    Command line parser.
    '''
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Package ALOS L0 stripmap data into NISAR L0B HDF5")
    parser.add_argument('-i', '--indir', dest='indir', type=str,
                        help="Folder containing one ALOS L0 module",
                        required=True)
    parser.add_argument('-o', '--outh5', dest='outh5', type=str,
                        help="Name of output file. If not provided, will be determined from ALOS granule",
                        default=None)
    parser.add_argument('-d', '--debug', dest='deug', action='store_true',
                        help="Use more rigorous parser to check magic bytes aggresively",
                        default=False)

    inps = parser.parse_args()
    if not os.path.isdir(inps.indir):
        raise ValueError('{0} does not appear to be a directory'.format(inps.indir))

    if inps.outh5 is None:
        print('HDF5 output granule name will be determined on fly and created in cwd')

    return inps

def getALOSFilenames(indir):
    '''
    Parse the contents of a given directory to separate out leader and image files.
    '''
    import glob
    import os

    filenames = {}

    ##First look for the leader file
    flist = glob.glob(os.path.join(indir, 'LED-ALPSRP*1.0__*'))
    if len(flist) == 0:
        raise ValueError('No leader files found in folder {0}'.format(indir))
    elif len(flist) > 1:
        raise ValueError('Multiple leader files in folder {0}'.format(indir))

    filenames['leaderfile']  = flist[0]
    pattern = os.path.basename(flist[0])[4:]


    ##Look for polarizations
    for pol in ['HH', 'HV', 'VV', 'VH']:
        flist = glob.glob(os.path.join(indir, 'IMG-{0}-{1}'.format(pol, pattern)))
        if len(flist) == 1:
            print('Found Polarization: {0}'.format(pol))
            filenames[pol] = flist[0]


    #If no image files were found
    if len(filenames) == 1:
        raise ValueError('No image files were found in folder: {0}'.format(indir))

    filenames['defaulth5'] = '{0}.h5'.format(pattern)

    return filenames


def parseLeaderFile(filenames):
    '''
    Parse leader file and check values against polarizations.
    '''

    from isce3.stripmap.readers.l0raw.ALOS.CEOS import LeaderFile
    try:
        ldr = LeaderFile.LeaderFile(filenames['leaderfile'])
    except AssertionError as msg:
        print(msg)
        raise AssertionError('Error parsing ALOS raw leader file: {0}'.format(filenames['leaderfile']))


    ##Checks to ensure that the number of polarizations is consistent
    numpol = len(filenames) - 2 #Subtract leader and defaulth5 name
    if numpol != ldr.summary.NumberOfSARChannels:
        raise ValueError('Number of image files discovered is inconsistent with Leader File')

    return ldr


def constructNISARHDF5(args, ldr):
    '''
    Build skeleton of HDF5 file using leader file information.
    '''
    import h5py
    import numpy
    import os
    import datetime

    #Open file for writing
    fid = h5py.File(args.outh5, 'w-')
    lsar = fid.create_group('/science/LSAR')

    ##Fill up Identification
    ident = lsar.create_group('identification')
    ident.create_dataset('diagnosticModeFlag', data=numpy.string_("False"))
    ident.create_dataset('isGeocoded', data=numpy.string_("True"))
    ident.create_dataset('listOfFrequencies', data=numpy.string_(["A"]))
    ident.create_dataset('lookDirection', data = numpy.string_("Right"))
    ident.create_dataset('missionId', data=numpy.string_("ALOS"))
    ident.create_dataset('orbitPassDirection', data=numpy.string_(ldr.summary.TimeDirectionIndicatorAlongLine))
    ident.create_dataset('processingType', data=numpy.string_("repackaging"))
    ident.create_dataset('productType', data=numpy.string_("RRSD"))
    ident.create_dataset('productVersion', data=numpy.string_("0.1"))


    ##Start populating metadata parts
    rrsd = lsar.create_group('RRSD')
    inps = rrsd.create_group('metadata/processingInformation/inputs')
    inps.create_dataset('l0aGranules', data=numpy.string_([os.path.basename(args.indir)]))

    #Start populating telemetry
    orbit = rrsd.create_group('telemetry/orbit')
    orbit.create_dataset('orbitType', data=numpy.string_('DOE'))
    tref = datetime.datetime(ldr.platformPosition.header.YearOfDataPoint,
                           ldr.platformPosition.header.MonthOfDataPoint,
                           ldr.platformPosition.header.DayOfDataPoint)

    t0 = ldr.platformPosition.header.SecondsOfDay
    dt = ldr.platformPosition.header.TimeIntervalBetweenDataPointsInSec

    pos = []
    vel = []
    times = []

    for ind, sv in enumerate(ldr.platformPosition.statevectors):
        times.append(t0 + ind * dt * 1.0)
        pos.append([sv.PositionXInm, sv.PositionYInm, sv.PositionZInm])
        vel.append([sv.VelocityXInmpers, sv.VelocityYInmpers, sv.VelocityZInmpers])

    time = orbit.create_dataset('time', data=numpy.array(times))
    time.attrs['units'] = "seconds since {0}".format(tref.strftime('%Y-%m-%d 00:00:00'))
    orbit.create_dataset('position', data=numpy.array(pos))
    orbit.create_dataset('velocity', data=numpy.array(vel))


    #Close the file
    fid.close()

def addImagery(h5file, ldr, imgfile, pol):
    '''
    Populate swaths segment of HDF5 file.
    '''
    import datetime
    import h5py
    import numpy
    import os
    from isce3.stripmap.readers.l0raw.ALOS.CEOS import ImageFile

    #Speed of light - expose in isce3
    SOL = 299792458.0

    fid = h5py.File(h5file, 'r+')
    assert(len(pol) == 2)
    txP = pol[0]
    rxP = pol[1]


    #Parse imagefile descriptor and first record.
    image = ImageFile.ImageFile(imgfile)
    firstrec = image.readNextLine()


    #Range related parameters
    fsamp = ldr.summary.SamplingRateInMHz * 1.0e6
    r0 = firstrec.SlantRangeToFirstSampleInm
    dr = SOL / (2 * fsamp)
    nPixels = image.description.NumberOfBytesOfSARDataPerRecord // image.description.NumberOfSamplesPerDataGroup
    nLines = image.description.NumberOfSARDataRecords


    freqA = '/science/LSAR/RRSD/swaths/frequencyA'

    #If this is first pol being written, add common information as well
    if freqA not in fid:
        freqA = fid.create_group(freqA)
        freqA.create_dataset('centerFrequency', data=SOL / (ldr.summary.RadarWavelengthInm))
        freqA.create_dataset('rangeBandwidth', data=ldr.calibration.header.BandwidthInMHz * 1.0e6)
        freqA.create_dataset('chirpDuration', data=firstrec.ChirpLengthInns * 1.0e-9)
        freqA.create_dataset('chirpSlope', data=-((freqA['rangeBandwidth'][()]*1.0e6)/(freqA['chirpDuration'][()]*1.0e-9)))
        freqA.create_dataset('nominalAcquisitionPRF', data=firstrec.PRFInmHz / 1000./ (1 + (ldr.summary.NumberOfSARChannels == 4)))
        freqA.create_dataset('slantRangeSpacing', data=dr)
        freqA.create_dataset('slantRange', data=r0 + numpy.arange(nPixels) * dr)
    else:
        freqA = fid[freqA]

        #Add bunch of assertions here if you want to be super sure that values are not different between pols


    ##Now add in transmit specific information
    txgrpstr = '/science/LSAR/RRSD/swaths/frequencyA/tx{0}'.format(txP)
    firstInPol = False
    if txgrpstr not in fid:
        firstInPol = True
        tstart = datetime.datetime(firstrec.SensorAcquisitionYear, 1, 1) +\
                 datetime.timedelta(days=int(firstrec.SensorAcquisitionDayOfYear-1))
        txgrp = fid.create_group(txgrpstr)
        time = txgrp.create_dataset('UTCTime', dtype='f', shape=(nLines,))
        time.attrs['units'] = "seconds since {0} 00:00:00".format(tstart.strftime('%Y-%m-%d'))
        txgrp.create_dataset('numberOfSubSwaths', data=1)
        txgrp.create_dataset('radarTime', dtype='f', shape=(nLines,))
        txgrp.create_dataset('rangeLineIndex', dtype='i8', shape=(nLines,))
        txgrp.create_dataset('validSamplesSubSwath1', dtype='i8', shape=(nLines,2))
    else:
        txgrp = fid[txgrpstr]

    ###Create imagery layer
    rximgstr = os.path.join(txgrpstr, 'rx{0}'.format(rxP))
    if rximgstr in fid:
        fid.close()
        raise ValueError('Reparsing polarization {0}. Array already exists {1}'.format(pol, rximgstr))

    print('Dimensions: {0}L x {1}P'.format(nLines, nPixels))

    cpxtype = numpy.dtype([('r', numpy.float16), ('i', numpy.float16)])
    fid.create_group(rximgstr)
    rximg = fid.create_dataset(os.path.join(rximgstr, pol), dtype=cpxtype, shape=(nLines,nPixels), chunks=True)

    ##Start populating the imagery
    bias = ldr.summary.DCBiasIComponent
    rec = firstrec
    for linnum in range(1, nLines+1):
        if (linnum % 1000 == 0):
            print('Parsing Line number: {0} out of {1}'.format(linnum, nLines))

        if firstInPol:
            txgrp['UTCTime'][linnum-1] = rec.SensorAcquisitionmsecsOfDay * 1.0e-3
            txgrp['rangeLineIndex'][linnum-1] = rec.SARImageDataLineNumber
            txgrp['radarTime'][linnum-1] = rec.SensorAcquisitionmsecsOfDay * 1.0e-3

        #Adjust range line
        rshift = int(numpy.rint((rec.SlantRangeToFirstSampleInm - r0) / dr))
        write_arr = numpy.full((2*nPixels), numpy.nan, dtype=numpy.float16)

        inarr = rec.SARRawSignalData[0,:].astype(numpy.float16)
        inarr[inarr == 0] = numpy.nan

        if rshift >= 0:
            write_arr[2*rshift:] = inarr[:2*(nPixels - rshift)] - bias
        else:
            write_arr[:2*rshift] = inarr[-2*rshift:] - bias

        if firstInPol:
            inds = numpy.where(~numpy.isnan(write_arr))
            if len(inds) > 1:
                txgrp['validSamplesSubSwath1'][linnum-1] = [inds[0], inds[-1]+1]

        #Complex float 16 writes work with write_direct only
        rximg.write_direct(write_arr.view(cpxtype), dest_sel=numpy.s_[linnum-1])

        ##Read next record
        if linnum != nLines:
            rec = image.readNextLine()


    if firstInPol:
        #Adjust time records - ALOS provides this only to nearest millisec - not good enough
        tinp = txgrp['UTCTime'][:]
        prf = freqA['nominalAcquisitionPRF'][()]
        tarr = (tinp - tinp[0]) * 1000
        ref = numpy.arange(tinp.size) / prf

        ####Check every 20 microsecs
        off = numpy.arange(-50,50)*2.0e-5
        res = numpy.zeros(off.size)

        ###Check which offset produces the same millisec truncation
        ###Assumes PRF is correct
        for xx in range(off.size):
            ttrunc = numpy.floor((ref+off[xx])*1000) #Should be round / floor?
            res[xx] = numpy.sum(tarr-ttrunc)

        delta = (numpy.argmin(numpy.abs(res)) - 50)*2.0e-5
        print('Start time correction in usec: ', delta*1e6)
        txgrp['UTCTime'][:] = ref + tinp[0] + delta

def process(args=None):
    '''
    Main processing workflow.
    '''
    import os

    #Discover file names
    filenames = getALOSFilenames(args.indir)

    #Parse the leader file
    leader = parseLeaderFile(filenames)

    #Set up output file name
    if args.outh5 is None:
        args.outh5 = filenames['defaulth5']

    if os.path.exists(args.outh5):
        raise ValueError('Output HDF5 file {0} already exists. Exiting ...'.format(args.outh5))


    #Setup HDF5 skeleton
    constructNISARHDF5(args, leader)


    #Iterate over polarizations for imagery layers
    for pol in ['HH', 'HV', 'VV', 'VH']:
        if pol in filenames:
            addImagery(args.outh5, leader, filenames[pol], pol)


if __name__ == "__main__":
    '''
    Main driver.
    '''

    #Parse command line
    inps = cmdLineParse()

    #Process the data
    process(args=inps)
