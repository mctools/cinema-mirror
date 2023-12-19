#!/bin/bash
#prerequest: install docker engine, see https://docs.docker.com/engine/install/ubuntu/ (note: linux distribution specific)
#run docker
# sudo docker run -i -t -v `pwd`:/io quay.io/pypa/manylinux2014_x86_64 /bin/bash

echo ${PREFIX} | tr -d '"'
project_name="neutron-cinema"

# --install wget
yes | yum install wget

# --install required applications hdf5 & xerces
cd /io
rm -rf hdf5-1.12.2
rm -rf hdf5-1.12.2.tar.gz*
wget https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-1.12/hdf5-1.12.2/src/hdf5-1.12.2.tar.gz
tar xvf hdf5-1.12.2.tar.gz
cd ./hdf5-1.12.2

./configure --prefix="/usr/local/hdf5"
make -j8
make install


# --install required development packages
yum -y install centos-release-scl-rh
yes | yum install rh-python38-python-devel
yes | yum install python3-devel
yes | yum install fftw-devel

export PATH="/usr/local/hdf5:/io/hdf5-1.12.2/:$PATH"

# --install ncrystal data
cd /io
rm -rf data_ncrystal
mkdir data_ncrystal
cd data_ncrystal
git init
git remote add -f origin ${PREFIX}/ncrystal.git
git config core.sparsecheckout true
git sparse-checkout set data
git pull origin master

# --remove pre-exist intermediate files
cd /io
rm -rf ./build/
rm -rf ./cinemabin/
rm -rf ./cinemavirenv/
rm -rf ./${project_name}.egg-info
rm -rf ./dist
rm -rf ./wheelrepaired

cd /io
. install.sh -f
git config --global --add safe.directory /io
for PYBIN in /opt/python/cp*/bin; do
    ## resolve local requirements
    #"${PYBIN}/python" install -r /io/dev-requirements.txt
    "${PYBIN}/python" setup.py bdist_wheel
done
# /opt/python/cp38-cp38/bin/python setup.py bdist_wheel
cd /io/dist
for WHL in *.whl; do
    auditwheel repair ${WHL} -w /io/wheelrepaired
done

cd /io