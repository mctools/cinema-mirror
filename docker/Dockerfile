# syntax=docker/dockerfile:1
#FROM arm64v8/ubuntu:20.04
FROM ubuntu:20.04 AS builder
LABEL maintainer="TANG Ming"

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive TZ="Asia/Shanghai" apt-get install -y \
    cmake \
    git \
    gcc-9 g++-9 \
    libfftw3-dev \
    libhdf5-cpp-103 libhdf5-dev \
    libxerces-c-dev \
    python3 python3-pip \
    #quantum-espresso \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

COPY ./cinema /cinema
ENV PIXIUSSSP="/cinema/pixiusssp"
ENV PATH="/cinema/external/ncrystal/install/bin:$PATH:/cinema/cinemabin:/cinema/cinemabin/bin:/cinema/scripts"
ENV LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/cinema/external/ncrystal/install/lib:/cinema/cinemabin:/cinema/src/python/bin"
ENV DYLD_LIBRARY_PATH="$DYLD_LIBRARY_PATH:/cinema/cinemabin"
ENV PYTHONPATH="/cinema/src/python:/cinema/external/ncrystal/install/share/NCrystal/python:$PYTHONPATH"
ENV NCRYSTAL_DATA_PATH="/cinema/ncmat:/cinema/external/ncystal/install/share/Ncrystal/data"
ENV CMAKE_PREFIX_PATH="/cinema/external/ncrystal/install/lib/cmake:$CMAKE_PREFIX_PATH"
ENV NCRYSTAL_DATADIR="/cinema/external/ncrystal/install/share/NCrystal/data"
ENV NCRYSTALDIR="/cinema/external/ncrystal/install"
ENV CINEMAPATH="/cinema"
#ARG NUMCPU="$(grep ^cpu\\scores /proc/cpuinfo | uniq |  awk '{print $4}')"
ARG NUMCPU=8
WORKDIR /cinema/external
RUN git clone https://gitlab.com/xxcai1/ncrystal.git
WORKDIR /cinema/external/ncrystal/build
RUN cmake -DCMAKE_INSTALL_PREFIX=/cinema/external/ncrystal/install .. && make -j${NUMCPU} && make install
#RUN /cinema/external/ncrystal/install/bin/ncrystal-config --setup

WORKDIR /cinema/external
RUN git clone https://gitlab.com/xxcai1/VecGeom.git
WORKDIR /cinema/external/VecGeom/build
RUN cmake -DCMAKE_INSTALL_PREFIX=/cinema/external/VecGeom/install -DGDML=On -DUSE_NAVINDEX=On .. && make -j${NUMCPU} && make install

WORKDIR /cinema/cinemabin
RUN cmake .. && make -j${NUMCPU}

WORKDIR /cinema
RUN git clone https://gitlab.com/xxcai1/pixiusssp.git \
    && cd pixiusssp \
    && mkdir SSSP_precision_pseudos \
    && tar -xzvf SSSP_1.1.2_PBE_precision.tar.gz --directory SSSP_precision_pseudos \
    && cp Cu.pbe-dn-kjpaw_psl.1.0.0.UPF SSSP_precision_pseudos/ \
    && rm SSSP_precision_pseudos/Cu_ONCV_PBE-1.0.oncvpsp.upf

RUN sed -i "s/pyvistaqt//g;s/pyvista//g;s/matplotlib//g;s/pymongo//g" /cinema/requirement
#RUN pip install --no-cache-dir -i https://mirrors.163.com/pypi/simple/ -r /cinema/requirement

RUN echo >/cinema/src/python/Cinema/Prompt/__init__.py

WORKDIR /cinema/temp
RUN cp /cinema/requirement .
RUN cp -R /cinema/ncmat .
RUN cp -R /cinema/scripts .
RUN cp -R /cinema/data .

WORKDIR /cinema/temp/src
RUN cp -R /cinema/src/python .
RUN cp -R /cinema/src/pythontests .

RUN find /cinema/temp -name *.pyc -delete
RUN find /cinema/temp -type d -name __pycache__ -delete

WORKDIR /cinema/temp/external/ncrystal
RUN cp -R /cinema/external/ncrystal/install .

WORKDIR /cinema/temp/external/VecGeom
RUN cp -R /cinema/external/VecGeom/install .

WORKDIR /cinema/temp/cinemabin/src/cxx
RUN cp /cinema/cinemabin/src/cxx/libprompt_core.so .
WORKDIR /cinema/temp/cinemabin/bin
RUN cp /cinema/cinemabin/bin/trj2sqw .
RUN cp /cinema/cinemabin/bin/prompt .

WORKDIR /cinema/temp/external/pixiusssp
RUN cp -R /cinema/pixiusssp/SSSP_precision_pseudos . && cp /cinema/pixiusssp/SSSP_1.2.0_PBE_precision.json .

#RUN cp /cinema/temp /cinema

WORKDIR /cinema
CMD ["bash"]

####################################
FROM ubuntu:20.04
LABEL maintainer="TANG Ming"

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive TZ="Asia/Shanghai" apt-get install -y \
    libfftw3-dev \
    libhdf5-cpp-103 libhdf5-dev \
    libxerces-c-dev \
    python3 python3-pip \
    quantum-espresso \
    #locales \
    #&& localedef -i en_US -c -f UTF-8 -A /usr/share/locale.alias en_US.UTF-8 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

COPY --from=builder /cinema/temp /cinema
RUN pip install --no-cache-dir -i https://mirrors.163.com/pypi/simple/ -r /cinema/requirement
ENV PIXIUSSSP="/cinema/external/pixiusssp" \
    PATH="/cinema/external/ncrystal/install/bin:$PATH:/cinema/cinemabin:/cinema/cinemabin/bin:/cinema/scripts" \
    LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/cinema/external/ncrystal/install/lib:/cinema/cinemabin:/cinema/src/python/bin" \
    DYLD_LIBRARY_PATH="$DYLD_LIBRARY_PATH:/cinema/cinemabin" \
    PYTHONPATH="/cinema/src/python:/cinema/external/ncrystal/install/share/NCrystal/python:$PYTHONPATH" \
    NCRYSTAL_DATA_PATH="/cinema/ncmat:/cinema/external/ncystal/install/share/Ncrystal/data" \
    NCRYSTAL_DATADIR="/cinema/external/ncrystal/install/share/NCrystal/data" \
    NCRYSTALDIR="/cinema/external/ncrystal/install" \
    CINEMAPATH="/cinema" \
    OMPI_MCA_btl_vader_single_copy_mechanism=none \
    OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1

#ENV LANG="en_US.utf8"
#ARG MY_USER="cinema"
#RUN adduser --disabled-password --home /home/${MY_USER} --gecos "" --shell /bin/bash ${MY_USER}
#USER ${MY_USER}
#ENV HOME="/home/${MY_USER}"
#WORKDIR "/home/${MY_USER}"

WORKDIR /data
CMD ["bash"]
