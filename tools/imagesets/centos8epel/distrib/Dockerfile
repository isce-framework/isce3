ARG runtime_img
FROM $runtime_img

# Extract ISCE3 installation to /usr
COPY isce3.rpm /
RUN rpm -i isce3.rpm
RUN rm isce3.rpm
