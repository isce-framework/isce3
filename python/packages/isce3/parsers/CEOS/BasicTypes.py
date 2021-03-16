#Basic types needed for handling CEOS format data
class StringType(object):
    '''
    Class to read in stream of bytes.
    '''

    def __init__(self, size):
        '''
        Constructor takes length of string as input.
        '''

        self.size = size
        self.data = None

    def dtype(self):
        '''
        Return numpy format string to read into.
        '''

        return 'a{}'.format(self.size)

    def fromfile(self, fid):
        '''
        Read one entry from file into data.
        '''
        import numpy

        self.data = numpy.fromfile(fid, dtype=self.dtype(), count=1)

    def assign(self, x):
        '''
        Assign value to field.
        '''
        self.data = x

    def value(self):
        '''
        Return value as string.
        '''

        return self.data[0].decode('utf-8').strip()

class BlankType(StringType):
    '''
    Class to read in void bytes.
    '''

    def dtype(self):
        '''
        Return numpy format string to read into.
        '''
        
        return 'V{}'.format(self.size)

    def value(self):
        '''
        Return None for blanks.
        '''
        
        return None

class IntegerType(StringType):
    '''
    Treat the string as an integer.
    '''

    def value(self):
        '''
        Read one integer from file.
        '''
        return int(super().value())

class FloatType(StringType):
    '''
    Treat the string as an integer.
    '''

    def value(self):
        '''
        Read one integer from file.
        '''
        return float(super().value())

class BinaryType(object):
    '''
    Class to read in binary data directly.
    '''

    def __init__(self, intype, count=1):
        '''
        Constructor takes length of string as input.
        '''
        import numpy

        self.intype = intype
        self.size = numpy.dtype(intype).itemsize
        self.count = count
        self.data = None

    def dtype(self):
        '''
        Return numpy format string to read into.
        '''

        if self.count != 1:
            return (self.intype, self.count)
        else:
            return self.intype

    def fromfile(self, fid):
        '''
        Read one entry from file into data.
        '''
        import numpy  
        self.data = numpy.fromfile(fid, dtype=self.dtype(), count=self.count)

    def assign(self, x):
        '''
        Assign value to field.
        '''
        self.data = x

    def value(self):
        '''
        Return value as string.
        '''
        if self.count == 1:
            return self.data[0]
        else:
            return self.data


class MultiType(object):
    '''
    Generic type.
    '''

    def __init__(self, inlist):
        '''
        Initiate type with dictionary.
        '''

        if not isinstance(inlist, list):
            raise Exception('Only lists are supported by CEOS parser.')

        self.mapping = inlist
        self.data = None


    def dtype(self):
        '''
        Return equivalent numpy dtype.
        '''
        import numpy 

        typelist = []
        for k, v in self.mapping:
            typelist.append((k, v.dtype()))

        return numpy.dtype(typelist)

    def assign(self, indata):
        for k, v in self.mapping:
            v.assign(indata[k])
        
        self.data = indata

    def fromfile(self, fid):
        '''
        Parse record from file.
        '''
        import numpy  
        self.data = numpy.fromfile(fid, dtype=self.dtype(), count=1)
       
        #Propagate to children
        for k, v in self.mapping:
                v.assign(self.data[k])

    def __getattr__(self, key):
        '''
        Return parsed value.
        '''
        for k, v in self.mapping:
            if k == key:
                if isinstance(v, MultiType):
                    return v
                else:
                    return v.value()

        raise Exception('MultiType record does not contain field named {}'.format(key))

