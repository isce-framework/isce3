ARG distrib_img
# create intermediate image to hide credentials
FROM alpine 

# install git
RUN apk add git

# add credentials on build
ARG GIT_OAUTH_TOKEN
RUN cd /opt \
 && git clone https://$GIT_OAUTH_TOKEN@github-fn.jpl.nasa.gov/NISAR-ADT/QualityAssurance \
 && git clone https://$GIT_OAUTH_TOKEN@github-fn.jpl.nasa.gov/NISAR-ADT/CFChecker \
 && git clone https://$GIT_OAUTH_TOKEN@github-fn.jpl.nasa.gov/NISAR-ADT/calTools \
 && cd /opt/QualityAssurance && git checkout cd455fd && rm -rf .git \
 && cd /opt/CFChecker && git checkout R1.1 && rm -rf .git \
 && cd /opt/calTools && git checkout R2 && rm -rf .git 

FROM $distrib_img

RUN conda install testfixtures scikit-image
RUN conda install cfunits --channel conda-forge

# copy the repo from the intermediate image
COPY --from=0 /opt/QualityAssurance /opt/QualityAssurance
COPY --from=0 /opt/CFChecker /opt/CFChecker
COPY --from=0 /opt/calTools /opt/calTools

# install 
RUN cd /opt/QualityAssurance \
 && python setup.py install \
 && cd /opt/CFChecker \
 && python setup.py install \
 && cd /opt/calTools \
 && python setup.py install
