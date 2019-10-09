# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# externals
from . import gsl


# the uniform probability distribution
class uniform:
    """
    Encapsulation of the uniform probability distribution
    """


    # interface
    def sample(self):
        """
        Sample the uniform distribution using a random value from {rng}
        """
        # get the value
        return gsl.uniform_sample(self.support, self.rng.rng)


    def density(self, x):
        """
        Compute the probability density of the uniform distribution at {x}
        """
        # get the value
        return gsl.uniform_density(self.support, x)


    # higher level support
    def vector(self, vector):
        """
        Fill {vector} with random values
        """
        # fill the vector
        gsl.uniform_vector(self.support, self.rng.rng, vector.data)
        # and return it
        return vector


    def matrix(self, matrix):
        """
        Fill {matrix} with random values
        """
        # fill the matrix
        gsl.uniform_matrix(self.support, self.rng.rng, matrix.data)
        # and return it
        return matrix


    # meta methods
    def __init__(self, support, rng, **kwds):
        super().__init__(**kwds)
        self.rng = rng
        self.support = support
        return


    # implementation details
    support = None

# the uniform probability distribution
class uniform_pos:
    """
    Encapsulation of the positive uniform probability distribution
    """


    # interface
    def sample(self):
        """
        Sample the uniform distribution using a random value from {rng}
        """
        # get the value
        return gsl.uniform_pos_sample(self.rng.rng)


    def density(self, x):
        """
        Compute the probability density of the uniform distribution at {x}
        """
        # get the value
        return 1.0


    # higher level support
    def vector(self, vector):
        """
        Fill {vector} with random values
        """
        # fill the vector
        gsl.uniform_pos_vector(self.rng.rng, vector.data)
        # and return it
        return vector


    def matrix(self, matrix):
        """
        Fill {matrix} with random values
        """
        # fill the matrix
        gsl.uniform_pos_matrix(self.rng.rng, matrix.data)
        # and return it
        return matrix


    # meta methods
    def __init__(self, rng, **kwds):
        super().__init__(**kwds)
        self.rng = rng
        return


# the gaussian probability distribution
class gaussian:
    """
    Encapsulation of the gaussian probability distribution
    """


    # interface
    def sample(self):
        """
        Sample the gaussian distribution using a random value from {rng}
        """
        # get the value
        return gsl.gaussian_sample(self.mean, self.sigma, self.rng.rng)


    def density(self, x):
        """
        Compute the probability density of the gaussian distribution at {x}
        """
        # get the value
        return gsl.gaussian_density(self.mean, self.sigma, x)


    # higher level support
    def vector(self, vector):
        """
        Fill {vector} with random values
        """
        # fill the vector
        gsl.gaussian_vector(self.mean, self.sigma, self.rng.rng, vector.data)
        # and return it
        return vector


    def matrix(self, matrix):
        """
        Fill {matrix} with random values
        """
        # fill the matrix
        gsl.gaussian_matrix(self.mean, self.sigma, self.rng.rng, matrix.data)
        # and return it
        return matrix


    # meta methods
    def __init__(self, mean, sigma, rng, **kwds):
        super().__init__(**kwds)
        self.rng = rng
        self.mean = mean
        self.sigma = sigma
        return


    # implementation details
    mean = 0.0
    sigma = None


# the unit gaussian probability distribution
class ugaussian:
    """
    Encapsulation of the unit gaussian probability distribution
    """


    # interface
    def sample(self):
        """
        Sample the gaussian distribution using a random value from {rng}
        """
        # get the value
        return gsl.ugaussian_sample(self.rng.rng)


    def density(self, x):
        """
        Compute the probability density of the gaussian distribution at {x}
        """
        # get the value
        return gsl.ugaussian_density(x)


    # higher level support
    def vector(self, vector):
        """
        Fill {vector} with random values
        """
        # fill the vector
        gsl.ugaussian_vector(self.rng.rng, vector.data)
        # and return it
        return vector


    def matrix(self, matrix):
        """
        Fill {matrix} with random values
        """
        # fill the matrix
        gsl.ugaussian_matrix(self.rng.rng, matrix.data)
        # and return it
        return matrix


    # meta methods
    def __init__(self, rng, **kwds):
        super().__init__(**kwds)
        self.rng = rng
        return


# the dirichlet probability distribution
class dirichlet:
    """
    Encapsulation of the dirichlet probability distribution
    """

    # higher level support
    def vector(self, vector):
        """
        Fill {vector} with random values
        """
        # fill the vector
        gsl.dirichlet_vector(self.rng.rng, self.alpha.data, vector.data)
        # and return it
        return vector


    def matrix(self, matrix):
        """
        Fill {matrix} with random values
        """
        # fill the matrix
        gsl.dirichlet_matrix(self.rng.rng, self.alpha.data, matrix.data)
        # and return it
        return matrix


    # meta methods
    def __init__(self, alpha, rng, **kwds):
        super().__init__(**kwds)
        self.rng = rng
        self.alpha = alpha
        return


# end of file
