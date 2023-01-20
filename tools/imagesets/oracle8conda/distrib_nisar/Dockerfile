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
 && cd /opt/QualityAssurance && git checkout 0ed82e7 && rm -rf .git \
 && cd /opt/CFChecker && git checkout R2 && rm -rf .git \
 && cd /opt/SoilMoisture && git checkout 6db3910852e7488d51bfc38366c5f8bae79cc3e9 && rm -rf .git

FROM $distrib_img

RUN conda install testfixtures scikit-image --channel conda-forge
RUN conda install cfunits --channel conda-forge

# Soil Moisture
COPY soilmoisture-spec-file.txt /tmp/spec-file.txt
RUN conda create -n SoilMoisture --file /tmp/spec-file.txt && conda clean -ay

# copy the repo from the intermediate image
COPY --from=0 /opt/QualityAssurance /opt/QualityAssurance
COPY --from=0 /opt/CFChecker /opt/CFChecker
COPY --from=0 /opt/SoilMoisture /opt/SoilMoisture

# install 
RUN cd /opt/QualityAssurance && python setup.py install
RUN cd /opt/CFChecker && python setup.py install
RUN cd /opt/SoilMoisture/src/release/r32 && conda run -n SoilMoisture make install