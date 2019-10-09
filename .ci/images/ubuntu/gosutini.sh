export PATH="/opt/conda/bin:${PATH}" && \
conda install --yes gosu && \
CONDA_GOSU_INFO=`conda list gosu | grep gosu` && \
echo "gosu ${CONDA_GOSU_INFO[1]}" >> /opt/conda/conda-meta/pinned && \
conda install --yes tini && \
CONDA_TINI_INFO=`conda list tini | grep tini` && \
echo "tini ${CONDA_TINI_INFO[1]}" >> /opt/conda/conda-meta/pinned && \
 conda clean -tipsy
