ARG runtime_img
FROM $runtime_img

# Extract ISCE3 installation to /usr
COPY isce3.rpm /
# XXX dependencies from conda env not detected correctly
RUN rpm -i isce3.rpm --nodeps \
 && echo /usr/local/lib64 >> /etc/ld.so.conf.d/isce3.conf && ldconfig
RUN rm isce3.rpm
