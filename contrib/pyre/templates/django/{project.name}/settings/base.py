# -*- Python -*-
# -*- coding: utf-8 -*-
#
# {project.authors}
# {project.affiliations}
# (c) {project.span} all rights reserved
#

"""
Django settings for {project.name}

For more information on this file, see
https://docs.djangoproject.com/en/1.7/topics/settings/

For the full list of settings and their values, see
https://docs.djangoproject.com/en/1.7/ref/settings/
"""

# python imports
import os
import {project.name}

# folder definitions
BASE = os.path.abspath(os.path.join({project.name}.home, os.pardir, os.pardir))
WEB = os.path.join(BASE, 'web', 'www', '{project.name}')
# important folder definitions
TEMPLATES = os.path.join(WEB, 'templates')
RESOURCES = os.path.join(WEB, 'resources')
UPLOADS = os.path.join(RESOURCES, 'uploads')

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = 'c_j=+(4w66f+w0z22j$oapix_974gnq8@9+r4)hjf0gea%qd1j'

# WSGI_APPLICATION = 'apache.wsgi.application'

ALLOWED_HOSTS = []

# Internationalization
# https://docs.djangoproject.com/en/1.7/topics/i18n/

LANGUAGE_CODE = 'en-us'

TIME_ZONE = 'UTC'

USE_I18N = True

USE_L10N = True

USE_TZ = True

# end of file
