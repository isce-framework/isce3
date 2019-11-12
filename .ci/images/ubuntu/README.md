# Centos Base Image

This is a simple centos base image that installs gcc6 + python3 via anaconda.

1. Python requirements are captured in requirements.txt
2. User named "conda" is the default user in the container
3. Container uses gosu and tini for better signal handling                         
