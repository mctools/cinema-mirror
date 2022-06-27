#!/bin/bash

echo "Build the image ..."
docker build -t cinema:0.1 -f Dockerfile ../..
#docker build --no-cache --progress=plain -t cinema:0.1 -f Dockerfile ../.. 2>&1 | tee build.log
