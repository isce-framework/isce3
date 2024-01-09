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
 && cd /opt/QualityAssurance && git checkout v7.0.1 && rm -rf .git \
 && cd /opt/SoilMoisture && git checkout 3f6f2f102cb0d187067a8a8b30bb6770d273b1f2 && rm -rf .git

FROM $distrib_img

# Soil Moisture
COPY soilmoisture-spec-file.txt /tmp/spec-file.txt
RUN conda create -n SoilMoisture --file /tmp/spec-file.txt && conda clean -ay

# copy the repo from the intermediate image
COPY --from=0 /opt/QualityAssurance /opt/QualityAssurance
COPY --from=0 /opt/SoilMoisture /opt/SoilMoisture

# install
RUN cd /opt/QualityAssurance && pip install --no-deps .
RUN cd /opt/SoilMoisture/src && conda run -n SoilMoisture make install

# Add image labels following Open Containers Initiative specifications
# (https://github.com/opencontainers/image-spec/blob/main/annotations.md)
ARG CREATION_DATETIME
ARG ISCE3_VERSION
ARG GIT_COMMIT
LABEL "org.opencontainers.image.created"="${CREATION_DATETIME}"
LABEL "org.opencontainers.image.version"="${ISCE3_VERSION}"
LABEL "org.opencontainers.image.revision"="${GIT_COMMIT}"
LABEL "org.opencontainers.image.url"="https://github.com/isce-framework/isce3"
LABEL "org.opencontainers.image.licenses"="Apache-2.0"
