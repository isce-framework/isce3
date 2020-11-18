ARG distrib_img
# create intermediate image to hide credentials
FROM $distrib_img as distrib_nisar_tmp

# install git
RUN conda install git

# add credentials on build
ARG GIT_OAUTH_TOKEN
RUN cd /opt \
 && git clone https://$GIT_OAUTH_TOKEN@github-fn.jpl.nasa.gov/NISAR-ADT/QualityAssurance \
 && git clone https://$GIT_OAUTH_TOKEN@github-fn.jpl.nasa.gov/NISAR-ADT/CFChecker

FROM $distrib_img

RUN conda install testfixtures \
 && conda install scikit-image

# copy the repo from the intermediate image
COPY --from=distrib_nisar_tmp /opt/QualityAssurance /opt/QualityAssurance
COPY --from=distrib_nisar_tmp /opt/CFChecker /opt/CFChecker

RUN cd /opt/QualityAssurance \
 && python setup.py install \
 && cd /opt/CFChecker \
 && python setup.py install 
