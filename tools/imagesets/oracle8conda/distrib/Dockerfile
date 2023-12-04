ARG runtime_img
FROM $runtime_img

# Extract ISCE3 installation to /usr
COPY isce3.rpm /
# XXX dependencies from conda env not detected correctly
RUN rpm -i isce3.rpm --nodeps \
 && echo /usr/local/lib64 >> /etc/ld.so.conf.d/isce3.conf && ldconfig
RUN rm isce3.rpm

# XXX Add a dummy config file for Herbie (a transitive dependency via RAiDER) which
# checks for this file in `~/.config/herbie/config.toml` when it's first imported
# and creates the file if it did not exist. This tends to cause permission errors in
# typical NISAR usage since generally users will not have a username or home directory
# inside the container image (`$HOME` expands to `/`).
# Relevant issues:
# - https://github-fn.jpl.nasa.gov/isce-3/isce/issues/1564
# - https://github.com/dbekaert/RAiDER/issues/608
# - https://github.com/blaylockbk/Herbie/issues/243
COPY herbie-config.toml /.config/herbie/config.toml
