#!/bin/bash

docker run -it --rm -v $(pwd):/data cinema:0.1

# RUN with ptrace
#docker run --cap-add=SYS_PTRACE -it --rm -v $(pwd):/data cinema:0.1
