ARG distrib_img
# create intermediate image to hide credentials
FROM alpine 

# install git
RUN apk add git

# add credentials on build
ARG GIT_OAUTH_TOKEN
RUN cd /opt \
 && git clone https://$GIT_OAUTH_TOKEN@github-fn.jpl.nasa.gov/NISAR-ADT/SoilMoisture \
 && git clone https://$GIT_OAUTH_TOKEN@github-fn.jpl.nasa.gov/NISAR-ADT/QualityAssurance \
 && git clone https://$GIT_OAUTH_TOKEN@github-fn.jpl.nasa.gov/NISAR-ADT/CFChecker \
 && git clone https://$GIT_OAUTH_TOKEN@github-fn.jpl.nasa.gov/NISAR-ADT/calTools \
 && cd /opt/QualityAssurance && git checkout 448db8d && rm -rf .git \
 && cd /opt/CFChecker && git checkout R2 && rm -rf .git \
 && cd /opt/calTools && git checkout 5607f81 && rm -rf .git \
 && cd /opt/SoilMoisture && git checkout 80e14ac && rm -rf .git

FROM $distrib_img

RUN conda install testfixtures scikit-image
RUN conda install cfunits --channel conda-forge

# Soil Moisture
COPY spec-file.txt /tmp/spec-file.txt
RUN conda create -n SoilMoisture --file /tmp/spec-file.txt && conda clean -ay

# copy the repo from the intermediate image
COPY --from=0 /opt/QualityAssurance /opt/QualityAssurance
COPY --from=0 /opt/CFChecker /opt/CFChecker
COPY --from=0 /opt/calTools /opt/calTools
COPY --from=0 /opt/SoilMoisture /opt/SoilMoisture

# install 
RUN cd /opt/QualityAssurance && python setup.py install
RUN cd /opt/CFChecker && python setup.py install
RUN cd /opt/calTools && python setup.py install
RUN cd /opt/SoilMoisture && conda run -n SoilMoisture make install
