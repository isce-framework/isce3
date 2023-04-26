ARG distrib_img
# create intermediate image to hide credentials
FROM alpine 

# install git
RUN apk add git

# add credentials on build
# final:  git checkout hash
ARG GIT_OAUTH_TOKEN
RUN cd /opt \
 && git clone https://$GIT_OAUTH_TOKEN@github-fn.jpl.nasa.gov/NISAR-ADT/SoilMoisture \
 && git clone https://$GIT_OAUTH_TOKEN@github-fn.jpl.nasa.gov/NISAR-ADT/QualityAssurance \
 && cd /opt/QualityAssurance && git checkout v3.0.1 && rm -rf .git \
 && cd /opt/SoilMoisture && git checkout 9426d58ce14bc3051403b38099e9856f40697bca && rm -rf .git

FROM $distrib_img

# Soil Moisture
COPY soilmoisture-spec-file.txt /tmp/spec-file.txt
RUN conda create -n SoilMoisture --file /tmp/spec-file.txt && conda clean -ay

# copy the repo from the intermediate image
COPY --from=0 /opt/QualityAssurance /opt/QualityAssurance
COPY --from=0 /opt/SoilMoisture /opt/SoilMoisture

# install 
RUN cd /opt/QualityAssurance && python setup.py install
RUN cd /opt/SoilMoisture/src && conda run -n SoilMoisture make install
