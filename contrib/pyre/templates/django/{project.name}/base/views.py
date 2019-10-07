# -*- Python -*-
# -*- coding: utf-8 -*-
#
# {project.authors}
# {project.affiliations}
# (c) {project.span} all rights reserved
#

# this file describes the base views for {project.name}

# django imports
from django.shortcuts import render_to_response

# index view
def home(request):
    # return the rendered template
    return render_to_response('index.jade')


# end of file
