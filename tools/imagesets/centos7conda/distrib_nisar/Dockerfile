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
 && cd QualityAssurance && git checkout v1.1.0-ER && rm -rf .git && cd .. \
 && cd CFChecker && git checkout v1.1.0-ER && rm -rf .git && cd ..  

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
