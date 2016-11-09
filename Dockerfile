FROM nvidia/cuda:8.0-devel-ubuntu16.04

MAINTAINER Abdul Dakkak "dakkak@illinois.edu"

# Set one or more individual labels
LABEL com.wbgo.version="0.0.1"
LABEL vendor="ECE 408 Project"

#Build Essentials
RUN apt-get update && apt-get install --no-install-recommends -y \
      build-essential                        \
      bzip2                                  \
      tar                                    \
      unzip                                  \
      zlib1g-dbg                             \
      wget                                   \
      libopenmpi-dev                         \
      git                                    \
      file                                   \
      libhdf5-10                             \
      libhdf5-10-dbg                         \
      libhdf5-openmpi-10                     \
      libhdf5-openmpi-10-dbg                 \
      libhdf5-openmpi-dev                    \
      libhdf5-dev                            \
      libhdf5-cpp-11                         \
      libhdf5-cpp-11-dbg                     \
      cmake &&                               \
    rm -rf /var/lib/apt/lists/*

#Setup User
ENV USERNAME raiuser
RUN useradd -ms /bin/bash ${USERNAME}
ENV USERDIR /home/${USERNAME}

ENV HOME ${USERDIR}
ENV SRCDIR ${HOME}/src
ENV DATADIR ${HOME}/data
ENV BUILDDIR ${HOME}/build


COPY . ${SRCDIR}
COPY ./data ${DATADIR}

WORKDIR ${BUILDDIR}
RUN cmake -DCONFIG_USE_HUNTER=OFF ${SRCDIR}
RUN make

WORKDIR ${BUILDDIR}