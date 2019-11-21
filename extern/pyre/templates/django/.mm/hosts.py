# -*- coding: utf-8 -*-
#
#
# {project.authors}
# {project.affiliations}
# (c) {project.span} all rights reserved
#


# the builder decorator
def host(builder):
    """
    Decorate the builder with host specific options
    """
    # here is how you get host information
    host = builder.host
    # return the builder
    return builder


# end of file
