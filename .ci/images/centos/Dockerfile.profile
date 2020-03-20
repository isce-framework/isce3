FROM nisar/isce-ops:__TAG__

# Set an encoding to make things work smoothly.
ENV LANG en_US.UTF-8

# copy WorkflowProfile repo
COPY . /opt/WorkflowProfile

# Setup test directory   
RUN set -ex \
  && source /opt/docker/bin/entrypoint_source \
  && mkdir /home/conda/test \
  && cd /home/conda/test 

