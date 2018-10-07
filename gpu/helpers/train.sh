#!/bin/sh

CONTAINER_APP=/opt/oculomotor

nvidia-docker run -it --rm -v ${PWD}:${CONTAINER_APP} wbap/oculomotor python ${CONTAINER_APP}/application/train.py $*

