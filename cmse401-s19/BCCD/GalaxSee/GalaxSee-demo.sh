#!/bin/bash

# $Id: auto_time-live 3537 2012-01-06 04:51:48Z skylar $

set -eo pipefail

pushd ${HOME}/GalaxSee

. /usr/local/Modules/3.2.10/init/bash
module purge && module load modules mpich2
make -j3

bccd-syncdir --ni ${HOME}/GalaxSee ${HOME}/machines

bccd-snarfhosts
cp ${HOME}/machines $(pwd)
CORES=$(awk -F: '{s+=$2} END {print s}' $(pwd)/machines)

for((i=1;i<=$((${CORES}*2));i++)); do
	CMD="mpirun -np ${i} /tmp/$(hostname -s)-$(whoami)/GalaxSee 500 400 5000"
	echo "Running ${CMD}"
	echo "PROCS ${i}"
	time ${CMD}
done

popd
