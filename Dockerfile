FROM webgpu/rai:base

MAINTAINER Abdul Dakkak "dakkak@illinois.edu"

# Set one or more individual labels
LABEL com.webgpu.project.ece408.version="2016.0.1"
LABEL vendor="ECE 408 Project"

COPY . ${SRCDIR}
COPY ./data ${DATADIR}

WORKDIR ${BUILDDIR}
RUN cmake -DCONFIG_USE_HUNTER=OFF ${SRCDIR}
RUN make

WORKDIR ${BUILDDIR}
