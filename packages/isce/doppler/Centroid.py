#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#Author:
#Copyright 2010, by the California Institute of Technology. ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.
#Any commercial use must be negotiated with the Office of Technology Transfer at the California Institute of Technology.
#
#This software may be subject to U.S. export control laws. By accepting this software, the user agrees to comply with all applicable U.S.
#export laws and regulations. User has the responsibility to obtain export licenses, or other export authority as may be required before 
#exporting such information to foreign countries or providing access to foreign persons.
#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import isce
import logging
import math
import random
import isceobj
from isceobj.Scene.Frame import Frame
from iscesys.Component.Component import Component, Port
from isceobj.Util.mathModule import MathModule
import spisce

class Centroid(spisce.doppler.Centroid):

    logging_name = "isce.DopIQ"
    def __init__(self):
        super(DopIQ, self).__init__()
#        self.logger = logging.getLogger(
        self.rawImage = ''
        self.rawFilename = ''
        self.prf = None
        self.mean = None
        self.lineLength = None
        self.lineHeaderLength = None
        self.lastSample = None
        self.startLine = 1
        self.numberOfLines = None
        self.dim1_doppler = None     
        self.fractionalDoppler = []
        self.pixelIndex = []
        self.linear = {}
        self.quadratic = {}
        
#        self.createPorts()
        self.dictionaryOfVariables = {'PRF': ['self.prf','float','mandatory'],                                      
                                      'I_BIAS': ['self.mean','float','mandatory'],
                                      'WIDTH': ['self.lineLength','int','mandatory'],
                                      'XMIN': ['self.lineHeaderLength','int','mandatory'],
                                      'XMAX': ['self.lastSample','int','mandatory'],
                                      'YMIN': ['self.startLine','int','mandatory'],
                                      'FILE_LENGTH': ['self.numberOfLines','int','mandatory']}
                                      
        self.dictionaryOfOutputVariables= {'DOPPLER': 'self.fractionalDoppler'}
        self.descriptionOfVariables = {}
        
        return None

    def createPorts(self):
        # Create Input Ports
        instrumentPort = Port(name="instrument",method=self.addInstrument,
                              doc="An object that has getPulseRepetitionFrequency() and getInPhaseValue() methods")
        framePort = Port(name="frame",method=self.addFrame,
                         doc="An object that has getNumberOfSamples() and getNumberOfLines() methods")
        imagePort = Port(name="image",method=self.addImage,
                         doc="An object that has getXmin() and getXmax() methods")                
        self._inputPorts.add(instrumentPort)
        self._inputPorts.add(framePort)
        self._inputPorts.add(imagePort)
        

    def addInstrument(self):
        instrument = self._inputPorts.getPort('instrument').getObject()
        if (instrument):
            try:            
                self.prf = instrument.getPulseRepetitionFrequency()
                self.mean = instrument.getInPhaseValue()
            except AttributeError:
                self.logger.error("Object %s requires a getPulseRepetitionFrequency() and getInPhaseValue() method" % (instrument.__class__))
        
    def addFrame(self):
        frame = self._inputPorts.getPort('frame').getObject()
        if(frame):
            try:                                
                self.numberOfLines = frame.getNumberOfLines()
            except AttributeError:
                self.logger.error("Object %s requires a getNumberOfSamples() and getNumberOfLines() method" % (frame.__class__))
        
    def addImage(self):
        image = self._inputPorts.getPort('image').getObject()
        if(image):
            try:
                self.rawFilename = image.getFilename()
                self.lineHeaderLength = image.getXmin()
                self.lastSample = image.getXmax()
                self.lineLength = self.lastSample
            except AttributeError:
                self.logger.error("Object %s requires getXmin(), getXmax() and getFilename() methods" % (image.__class__))                    

    def setRawfilename(self,filename):
        self.rawFilename = filename

    def setPRF(self,prf):
        self.prf = float(prf)

    def setMean(self,mean):
        self.mean = float(mean)

    def setLineLength(self,length):
        self.lineLength = int(length)

    def setLineHeaderLength(self,length):
        self.lineHeaderLength = int(length)

    def setLastSample(self,length):
        self.lastSample = int(length)

    def setNumberOfLines(self,lines):
        self.numberOfLines = int(lines)

    def setStartLine(self,start):
        if (start < 1):
            raise ValueError("START_LINE must be greater than 0")
        self.startLine = int(start)

    ##
    # Return the doppler estimates in Hz/prf as a function of range bin
    def getDoppler(self):        
        return self.fractionalDoppler


    @pyre.export
    def estimate(self, rawImage=None):
        self.activateInputPorts()

        rawCreatedHere = False
        if rawImage is None:
            self.rawImage = self.createRawImage()
            rawCreateHere = True
        else:
            self.rawImage = rawImage
            pass
        rawAccessor = self.rawImage.getImagePointer()
        self.setDefaults()
        self.rngDoppler = [0]*int((self.width - self.header)/2)
        self.allocateArrays()
        self.setState()
        spisce.isce.extensions.isce.estimate_dop(rawAccessor)
        self.getState()
        self.deallocateArrays()
        if rawCreatedHere:
            self.rawImage.finalizeImage()
            pass
        return None
          
    def createRawImage(self):
        # Check file name
        width = self.lineLength        
        objRaw = isceobj.createRawImage()
        objRaw.initImage(self.rawFilename,'read',width)
        objRaw.createImage()                
        return objRaw
    
    def setState(self):
        # Set up the stuff needed for dopiq
        dopiq.setPRF_Py(self.prf)
        dopiq.setNumberOfLines_Py(self.numberOfLines)
        dopiq.setMean_Py(self.mean)
        dopiq.setLineLength_Py(self.lineLength)
        dopiq.setLineHeaderLength_Py(self.lineHeaderLength)
        dopiq.setLastSample_Py(self.lastSample)
        dopiq.setStartLine_Py(self.startLine)
        self.dim1_doppler = int((self.lastSample - self.lineHeaderLength)/2)
        
    def getState(self):
        self.fractionalDoppler = dopiq.getDoppler_Py(self.dim1_doppler)        
        
    def allocateArrays(self):
        if (self.dim1_doppler == None):
            self.dim1_doppler = len(self.fractionalDoppler)
            
        if (not self.dim1_doppler):
            self.logger.error("Error. Trying to allocate zero size array")

            raise Exception
        
        dopiq.allocate_doppler_Py(self.dim1_doppler)
        
    def deallocateArrays(self):
        dopiq.deallocate_doppler_Py()

    def _wrap(self):
        """Wrap the Doppler values"""
        wrapCount = 0*5;
        noiseLevel = 0*0.7;
 
        for i in range(len(self.fractionalDoppler)):
            if ( wrapCount != 0 ):
                self.fractionalDoppler[i] += wrapCount + i * wrapCount / len(self.fractionalDoppler)

            if( noiseLevel != 0 ):
                self.fractionalDoppler[i] += 1 + noiseLevel/2 - random.random(noiseLevel)

            self.fractionalDoppler[i] -= int(self.fractionalDoppler[i])
 
    def _unwrap(self):
        """Unwrapping"""
         
        averageLength=10
        firstDop = 0
         
        lastValues = []
        unw = [None]*len(self.fractionalDoppler)
         
        for i in range(averageLength-1):
            lastValues.append(firstDop)
         
        for i in range(len(self.fractionalDoppler)):
            predicted = sum(lastValues) / len(lastValues)
            ambiguity = predicted - self.fractionalDoppler[i]
            ambiguity = int(ambiguity)
            unw[i] = self.fractionalDoppler[i] + ambiguity
         
        if ( len(lastValues) >= (averageLength-1)):
            lastValues.pop(0)
        lastValues.append(unw[i])
         
        return unw
         
    def _cullPoints(self,pixels,unw):
        """Remove points greater than 3 standard deviations from the line fit"""

        slope = self.linear['b']
        intercept = self.linear['a']
        stdDev = self.linear['stdDev'] 
        numCulled = 0
        newPixel = []
        newUnw = []
 
        for i in range(len(pixels)):	
            fit      = intercept + slope * pixels[i]
            residual = unw[i] - fit
            if ( math.fabs(residual) < 3*stdDev ):
                newPixel.append(pixels[i])
                newUnw.append(unw[i])
            else:
                numCulled += 1
         
        return newPixel, newUnw
 
    @pyre.export
    def unwrap(self):
        """
        Unwrap the doppler estimates
        """
        self._wrap()
        self.unw = self._unwrap()

    @pyre.export
    def parameterize(self):
        """
        Read in Doppler estimates, remove outliers and then perform a quadratic fit
        """
        self.pixelIndex = range(len(self.fractionalDoppler))
        (self.linear['a'], self.linear['b'], self.linear['stdDev']) = MathModule.linearFit(self.pixelIndex, self.unw)        
        (pixels, self.unw) = self._cullPoints(self.pixelIndex,self.unw)
        (self.linear['a'], self.linear['b'], self.linear['stdDev']) = MathModule.linearFit(pixels, self.unw)        
        (pixels, self.unw) = self._cullPoints(pixels,self.unw)
  
        (a,b,c) = MathModule.quadraticFit(pixels,self.unw)
        self.quadratic['a'] = a  
        self.quadratic['b'] = b  
        self.quadratic['c'] = c
