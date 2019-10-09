# -*- Python -*-
# -*- coding: utf-8 -*-
#
# {project.authors}
# {project.affiliations}
# (c) {project.span} all rights reserved
#

# this file contains the settings necessary for local deployment of {project.name}

from .{project.name} import *

# enable debugging support
DEBUG = True
TEMPLATE_DEBUG = True
COMPRESS_DEBUG_TOGGLE = True

# Database
# https://docs.djangoproject.com/en/1.7/ref/settings/#databases
DATABASES = {{
    'default': {{
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': os.path.join(BASE, 'var', '{project.name}', '{project.name}.sqlite3'),
    }}
}}

# end of file
