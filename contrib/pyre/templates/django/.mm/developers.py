# -*- coding: utf-8 -*-
#
# {project.authors}
# {project.affiliations}
# (c) {project.span} all rights reserved
#


def developer(builder):
    """
    Decorate the builder with developer specific choices
    """
    # here is how you get the developer name
    name = builder.user.name
    # return the builder
    return builder


# end of file
