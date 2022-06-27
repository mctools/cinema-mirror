# Docker

## Prerequisites
- Docker (Docker Engine && Docker CLI / Docker Desktop)

## Build
1. Clone Cinema

2. run
~~~Bash
cd cinema/docker
./build.sh
~~~

3. Or (for a fresh clean build)
~~~Bash
cd cinema/docker
docker build --no-cache -t cinema:0.1 -f Dockerfile ../..
~~~

Note: this step can be done only once, except you want to update to the latest Cinema.

## Run
~~~Bash
cd /PATH/TO/YOUR/WORKDIR
docker run -it --rm -v $(pwd):/data cinema:0.1
# Run the desired program
~~~
