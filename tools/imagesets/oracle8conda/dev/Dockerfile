ARG runtime_img
FROM $runtime_img

# Install dependencies for ISCE development
COPY spec-file.txt /tmp/spec-file.txt
RUN conda install --yes --file /tmp/spec-file.txt \
 && conda clean -ay \
 && rm -rf /opt/conda/pkgs \
 && rm /tmp/spec-file.txt

RUN mkdir -p /opt/conda/var/tmp \
 && chmod 777 /opt/conda/var/tmp

ENV CUDAHOSTCXX=x86_64-conda-linux-gnu-c++
