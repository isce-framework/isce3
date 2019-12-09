# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# externals
import numbers
import itertools
from . import gsl # the extension


# the class declaration
class Matrix:
    """
    A wrapper over a gsl matrix
    """

    # types
    from .Vector import Vector as vector

    # constants
    defaultFormat = "+16.7"

    upperTriangular = 1
    lowerTriangular = 0

    # flag that controls whether the diagonal entries are assumed to be unity
    unitDiagonal = 1
    nonUnitDiagonal = 0

    # operation flags for some of the blas primitives
    opNoTrans = 0
    opTrans = 1
    opConjTrans = 2

    # flag to control the order of operands in some matrix multiplication routines
    sideRight = 1
    sideLeft = 0

    # sort type for eigensystems
    sortValueAscending = 0
    sortValueDescending = 1
    sortMagnitudeAscending = 2
    sortMagnitudeDescending = 3


    # class methods
    # mpi support
    @classmethod
    def bcast(cls, matrix=None, communicator=None, source=0):
        """
        Broadcast the given {matrix} from {source} to all tasks in {communicator}
        """
        # normalize the communicator
        if communicator is None:
            # get the mpi package
            import mpi
            # use the world by default
            communicator = mpi.world
        # get the matrix capsule
        data = None if matrix is None else matrix.data
        # scatter the data
        capsule, shape = gsl.bcastMatrix(communicator.capsule, source, data)
        # dress up my local portion as a matrix
        result = cls(shape=shape, data=capsule)
        # and return it
        return result


    @classmethod
    def collect(cls, matrix, communicator=None, destination=0):
        """
        Gather the data in {matrix} from each task in {communicator} into one big matrix
        available at the {destination} task
        """
        # normalize the communicator
        if communicator is None:
            # get the mpi package
            import mpi
            # use the {world} by default
            communicator = mpi.world
        # gather the data
        result = gsl.gatherMatrix(communicator.capsule, destination, matrix.data)
        # if i am not the destination task, nothing further to do
        if communicator.rank != destination: return
        # otherwise, unpack the result
        data, shape = result
        # dress up the result as a matrix
        result = cls(shape=shape, data=data)
        # and return it
        return result


    def excerpt(self, communicator=None, source=0, matrix=None):
        """
        Scatter {matrix} held by the task {source} among all tasks in {communicator} and fill me
        with the partition values. Only {source} has to provide a {matrix}; the other tasks can
        use the default value.
        """
        # normalize the communicator
        if communicator is None:
            # get the mpi package
            import mpi
            # use the world by default
            communicator = mpi.world
        # get the matrix capsule
        data = None if matrix is None else matrix.data
        # scatter the data
        gsl.scatterMatrix(communicator.capsule, source, self.data, data)
        # and return me
        return self


    # public data
    @property
    def columns(self):
        """
        Get the number of columns
        """
        return self.shape[1]

    @property
    def rows(self):
        """
        Get the number of rows
        """
        return self.shape[0]

    @property
    def elements(self):
        """
        Iterate over all my elements
        """
        # i'm already accessible as an iterator
        yield from self
        # all done
        return


    # initialization
    def zero(self):
        """
        Set all my elements to zero
        """
        # zero me out
        gsl.matrix_zero(self.data)
        # and return
        return self


    def fill(self, value):
        """
        Set all my elements to {value}
        """
        # grab my capsule
        data = self.data
        # first, attempt to
        try:
            # convert value into a float
            value = float(value)
        # if this fails
        except TypeError:
            # go through the input values
            for idx, elem in zip(itertools.product(*map(range, self.shape)), value):
                # set each element
                gsl.matrix_set(data, idx, float(elem))
        # if the conversion to float were successful
        else:
            # fill
            gsl.matrix_fill(data, value)

        # all done
        return self


    def view(self, start, shape):
        """
        Build a view to my data anchored at {start} with the given {shape}
        """
        # access the view object
        from .MatrixView import MatrixView
        # build one and return it
        return MatrixView(matrix=self, start=start, shape=shape)


    def load(self, filename, binary=None):
        """
        Read my values from {filename}

        This method attempts to distinguish between text and binary representations of the
        data, based on the parameter {mode}, or the {filename} extension if {mode} is absent
        """
        # if the caller asked for binary mode
        if binary is True:
            # pick the binary representation
            return self.read(filename)

        # if the caller asked for ascii mode
        if binary is False:
            # pick ascii
            return self.scanf(filename)

        # otherwise, look at the file extension
        suffix = filename.suffix
        # if it's {bin}
        if suffix == "bin":
            # go binary
            return self.read(filename)

        # otherwise
        return self.scanf(filename)


    def save(self, filename, binary=None, format=defaultFormat):
        """
        Write my values to {filename}

        This method attempts to distinguish between text and binary representations of the
        data, based on the parameter {mode}, or the {filename} extension if {mode} is absent
        """
        # if the caller asked for binary mode
        if binary is True:
            # pick the binary representation
            return self.write(filename)

        # if the caller asked for ascii mode
        if binary is False:
            # pick ascii
            return self.printf(filename=filename, format=format)

        # otherwise, look at the file extension
        suffix = filename.suffix
        # if it's {bin}
        if suffix == ".bin":
            # go binary
            return self.write(filename)

        # otherwise
        return self.printf(filename=filename, format=format)


    def read(self, filename):
        """
        Read my values from {filename}
        """
        # read
        gsl.matrix_read(self.data, filename.path)
        # and return
        return self


    def write(self, filename):
        """
        Write my values to {filename}
        """
        # write
        gsl.matrix_write(self.data, filename.path)
        # and return
        return self


    def scanf(self, filename):
        """
        Read my values from {filename}
        """
        # read
        gsl.matrix_scanf(self.data, filename.path)
        # and return
        return self


    def printf(self, filename, format=defaultFormat):
        """
        Write my values to {filename}
        """
        # write
        gsl.matrix_printf(self.data, filename.path, '%'+format+'e')
        # and return
        return self


    def print(self, format='{:+13.4e}', indent='', interactive=True):
        """
        Print my values using the given {format}
        """
        # initialize the display
        lines = []
        # for each row
        for i in range(self.rows):
            # initialize the line
            fragments = []
            # print the left margin: a '[[' on the first row, nothing on the others
            fragments.append('{}{}'.format(indent, '[[' if i==0 else '  '))
            # the row entries
            for j in range(self.columns):
                fragments.append(format.format(self[i,j]))
            # the right margin
            fragments.append('{}'.format(']]' if i==self.rows-1 else '  '))
            # add the line to the pile
            lines.append(' '.join(fragments))

        # if we are in interactive mode
        if interactive:
            # print all this out
            print('\n'.join(lines))

        # all done
        return lines


    def identity(self):
        """
        Initialize me as an identity matrix: all elements are set to zero except along the
        diagonal, which are set to one
        """
        # initialize
        gsl.matrix_identity(self.data)
        # and return
        return self


    def random(self, pdf):
        """
        Fill me with random numbers using the probability distribution {pdf}
        """
        # the {pdf} knows how to do this
        return pdf.matrix(matrix=self)


    def clone(self):
        """
        Allocate a new matrix and initialize it using my values
        """
        # build the clone
        clone = type(self)(shape=self.shape)
        # have the extension initialize the clone
        gsl.matrix_copy(clone.data, self.data)
        # and return it
        return clone


    def copy(self, other):
        """
        Fill me with values from {other}, which is assumed to be of compatible shape
        """
        # fill me with values from {other}
        gsl.matrix_copy(self.data, other.data)
        # and return it
        return self


    def tuple(self):
        """
        Build a representation of my contents as a tuple of tuples

        This is suitable for converting to other matrix representations, such as numpy
        """
        # ask the extension to build the rep
        rep = gsl.matrix_tuple(self.data)
        # and return it
        return rep


    # matrix operations
    def transpose(self, destination=None):
        """
        Compute the transpose of a matrix.

        If {destination} is {None} and the matrix is square, the operation happens
        in-place. Otherwise, the transpose is stored in {destination}, which is assumed to be
        shaped correctly.
        """
        # if we have a {destination}
        if destination is not None:
            # do the transpose
            gsl.matrix_transpose(self.data, destination.data)
            # and return the destination matrix
            return destination
        # otherwise
        gsl.matrix_transpose(self.data, None)
        # and return myself
        return self


    # slicing
    def getRow(self, index):
        """
        Return a view to the requested row
        """
        # let the extension do its thing
        capsule = gsl.matrix_get_row(self.data, int(index))
        # build a vector and return it
        return self.vector(shape=self.columns, data=capsule)


    def getColumn(self, index):
        """
        Return a view to the requested column
        """
        # let the extension do its thing
        capsule = gsl.matrix_get_col(self.data, int(index))
        # build a vector and return it
        return self.vector(shape=self.rows, data=capsule)


    def setRow(self, index, v):
        """
        Set the row at {index} to the contents of the given vector {v}
        """
        # let the extension do its thing
        gsl.matrix_set_row(self.data, int(index), v.data)
        # and return
        return self


    def setColumn(self, index, v):
        """
        Set the column at {index} to the contents of the given vector {v}
        """
        # let the extension do its thing
        gsl.matrix_set_col(self.data, int(index), v.data)
        # and return
        return self


    # maxima and minima
    def max(self):
        """
        Compute my maximum value
        """
        # easy enough
        return gsl.matrix_max(self.data)


    def min(self):
        """
        Compute my maximum value
        """
        # easy enough
        return gsl.matrix_min(self.data)


    def minmax(self):
        """
        Compute my minimum and maximum values
        """
        # easy enough
        return gsl.matrix_minmax(self.data)


    # eigensystems
    def symmetricEigensystem(self, order=sortValueAscending):
        """
        Computed my eigenvalues and eigenvectors assuming i am a real symmetric matrix
        """
        # compute the eigenvalues and eigenvectors
        values, vectors = gsl.matrix_eigen_symmetric(self.data, order)
        # dress up the results
        λ = self.vector(shape=self.rows, data=values)
        x = type(self)(shape=self.shape, data=vectors)
        # and return
        return λ, x

    # statistics
    def mean(self, axis=None, out=None):
        """
        Compute the mean values of a matrix
        axis = None, 0, or 1, along which the mean are computed
        """
        # check axis
        if axis is not None and axis !=0 and axis !=1:
            raise IndexError("axis is out of range")
        # check whether output vector is already allocated
        if out is None:
            # mean, sd over flattened matrix
            if axis is None:
                mean = self.vector(shape=1)
            # mean, sd along row
            elif axis == 0:
                mean = self.vector(shape=self.columns)
            # mean, sd along column
            elif axis == 1:
                mean = self.vector(shape=self.rows)
        else:
            # use pre-allocated vectors
            mean = out
            # assuming correct dimension, skip error checking

        # call gsl function
        gsl.stats_matrix_mean(self.data, axis, mean.data)

        # return the result
        return mean

    def mean_sd(self, axis=None, out=None, sample=True):
        """
        Compute the mean values of matrix
        axis: int or None
             axis along which the means are computed. None for all elements
        out: tuple of two vectors (mean, sd)
             vector size is 1 (axis=None),  columns(axis=0), rows(axis=1)
        sample: True or False
             when True, the sample standard deviation is computed 1/(N-1)
             when False, the population standard deviation is computed 1/N
        """
        # check axis
        if axis is not None and axis !=0 and axis !=1:
            raise IndexError("axis is out of range")

        if out is None:
            # mean, sd over flattened matrix
            if axis is None:
                mean = self.vector(shape=1)
                sd = self.vector(shape=1)
            # mean, sd along row
            elif axis == 0:
                mean = self.vector(shape=self.columns)
                sd = self.vector(shape=self.columns)
            # mean, sd along column
            elif axis == 1:
                mean = self.vector(shape=self.rows)
                sd = self.vector(shape=self.rows)
        else:
            # use pre-allocated vectors
            mean, sd = out
            # assuming correct dimension, skip error checking

        # call gsl function
        if sample:
            gsl.stats_matrix_mean_sd(self.data, axis, mean.data, sd.data)
        else:
            gsl.stats_matrix_mean_std(self.data, axis, mean.data, sd.data)

        # return (mean, sd)
        return mean, sd

    def std(self, axis=None, sample=False):
        """
        Compute the standard deviation of a matrix
        """
        mean, sd = self.mean_sd(axis=axis, out=None, sample=sample)
        return sd


    def ndarray(self, copy=False):
        """
        Return a numpy array reference (w/ shared data) if {copy} is False, or a new copy if {copy} is {True}
        """
        # call c-api extension to create a numpy array reference
        array = gsl.matrix_ndarray(self.data)
        # whether the data copy is required
        if copy:
            array = array.copy()
        return array


    # meta methods
    def __init__(self, shape, data=None, **kwds):
        # chain up
        super().__init__(**kwds)
        # adjust the shape
        shape = tuple(map(int, shape))
        # store
        self.shape = shape
        # allocate
        self.data = gsl.matrix_alloc(shape) if data is None else data
        # all done
        return


    # container support
    def __iter__(self):
        """
        Iterate over all my elements in shape order
        """
        # unpack the shape
        index0, index1 = self.shape
        # go over all index pairs
        for index in itertools.product(*map(range, self.shape)):
            # grab the value
            yield gsl.matrix_get(self.data, index)
        # all done
        return


    def __contains__(self, value):
        # faster than checking every element in python
        return gsl.matrix_contains(self.data, value)


    def __getitem__(self, index):
        # get and return the element
        return gsl.matrix_get(self.data, index)


    def __setitem__(self, index, value):
        # set the element to the requested value
        return gsl.matrix_set(self.data, index, value)


    # comparisons
    def __eq__(self, other):
        # type check
        if type(self) is not type(other): return NotImplemented
        # hand the request off to the extension module
        return gsl.matrix_equal(self.data, other.data)


    def __ne__(self, other):
        return not (self == other)


    # in-place arithmetic
    def __iadd__(self, other):
        """
        In-place addition with the elements of {other}
        """
        # if other is a matrix
        if isinstance(other, Matrix):
            # do matrix-matrix addition
            gsl.matrix_add(self.data, other.data)
            # and return
            return self
        # if other is a number
        if isinstance(other, numbers.Number):
            # do constant addition
            gsl.matrix_shift(self.data, float(other))
            # and return
            return self
        # otherwise, let the interpreter know
        raise NotImplemented


    def __isub__(self, other):
        """
        In-place subtraction with the elements of {other}
        """
        # if other is a matrix
        if isinstance(other, Matrix):
            # do matrix-matrix subtraction
            gsl.matrix_sub(self.data, other.data)
            # and return
            return self
        # if other is a number
        if isinstance(other, numbers.Number):
            # do constant subtraction
            gsl.matrix_shift(self.data, -float(other))
            # and return
            return self
        # otherwise, let the interpreter know
        raise NotImplemented


    def __imul__(self, other):
        """
        In-place multiplication with the elements of {other}
        """
        # if other is a matrix
        if isinstance(other, Matrix):
            # do matrix-matrix multiplication
            gsl.matrix_mul(self.data, other.data)
            # and return
            return self
        # if other is a number
        if isinstance(other, numbers.Number):
            # do scaling by constant
            gsl.matrix_scale(self.data, float(other))
            # and return
            return self
        # otherwise, let the interpreter know
        raise NotImplemented


    def __itruediv__(self, other):
        """
        In-place addition with the elements of {other}
        """
        # if other is a matrix
        if isinstance(other, Matrix):
            # do matrix-matrix division
            gsl.matrix_div(self.data, other.data)
            # and return
            return self
        # if other is a number
        if isinstance(other, numbers.Number):
            # do scaling by constant
            gsl.matrix_scale(self.data, 1/float(other))
            # and return
            return self
        # otherwise, let the interpreter know
        raise NotImplemented


    # private data
    data = None
    shape = (0,0)


# end of file
