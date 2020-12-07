ARG distrib_img
# create intermediate image to hide credentials
FROM alpine as distrib_nisar_tmp

# install git
RUN apk add git

# add credentials on build
ARG GIT_OAUTH_TOKEN
RUN cd /opt \
 && git clone https://$GIT_OAUTH_TOKEN@github-fn.jpl.nasa.gov/NISAR-ADT/QualityAssurance \
 && git clone https://$GIT_OAUTH_TOKEN@github-fn.jpl.nasa.gov/NISAR-ADT/CFChecker \
 && git clone https://$GIT_OAUTH_TOKEN@github-fn.jpl.nasa.gov/NISAR-ADT/calTools \
 && cd QualityAssurance && git checkout 71e5f94a5fae37c3e0e7487c76baca3256035b95 && rm -rf .git && cd .. \
 && cd CFChecker && git checkout ce16a9ee55efba9a3be131a92e4c81a05e72323b && rm -rf .git && cd ..  

FROM $distrib_img

RUN conda install testfixtures scikit-image
RUN conda install cfunits --channel conda-forge

# copy the repo from the intermediate image
COPY --from=distrib_nisar_tmp /opt/QualityAssurance /opt/QualityAssurance
COPY --from=distrib_nisar_tmp /opt/CFChecker /opt/CFChecker

# install 
RUN cd /opt/QualityAssurance \
 && python setup.py install \
 && cd /opt/CFChecker \
 && python setup.py install
