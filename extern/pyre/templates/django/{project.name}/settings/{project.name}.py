# -*- Python -*-
# -*- coding: utf-8 -*-
#
# {project.authors}
# {project.affiliations}
# (c) {project.span} all rights reserved
#

# this file contains the settings that are specific to the application as a whole

from .base import *

# Application definition

django_apps = (
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
)

third_party_apps = (
    'compressor',
 )

{project.name}_apps = (
    '{project.name}.base',
)

INSTALLED_APPS = django_apps + third_party_apps + {project.name}_apps

MIDDLEWARE_CLASSES = (
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.auth.middleware.SessionAuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
)

ROOT_URLCONF = '{project.name}.urls'

APPEND_SLASH = True

# Static files (CSS, JavaScript, Images)

STATIC_URL = '/static/'
STATIC_ROOT = RESOURCES
MEDIA_ROOT = UPLOADS

STATICFILES_FINDERS = (
    'django.contrib.staticfiles.finders.FileSystemFinder',
    'django.contrib.staticfiles.finders.AppDirectoriesFinder',
    'compressor.finders.CompressorFinder',
)

# Template definitions

TEMPLATE_LOADERS = (
    ('pyjade.ext.django.Loader',(
        'django.template.loaders.filesystem.Loader',
        'django.template.loaders.app_directories.Loader',
    )),
)

TEMPLATE_DIRS = (
    TEMPLATES,
)

# django compressor settings

COMPRESS_ROOT = RESOURCES
COMPRESS_OUTPUT_DIR = "cache"

stylus_conf = ('-u jeet -u axis -u rupture -I ' +
               os.path.join(RESOURCES,'styles') +' < {{infile}} > {{outfile}}')

COMPRESS_PRECOMPILERS = (
    ('text/stylus', 'stylus '+ stylus_conf),
    ('text/coffeescript', 'coffee --compile --stdio -b'),
)

# end of file
