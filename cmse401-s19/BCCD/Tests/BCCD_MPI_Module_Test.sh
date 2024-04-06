#!/bin/bash

# $Id: mpitest.sh,v 1.1 2012/05/02 18:16:25 charliep Exp $

# This file is part of BCCD, an open-source live CD for computational science
# education.
# 
# Copyright (C) 2010 Andrew Fitz Gibbon, Paul Gray, Kevin Hunter, Dave Joiner, 
#   Sam Leeman-Munk, Tom Murphy, Charlie Peck, Skylar Thompson, & Aaron Weeden 

# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

if test -z "$1"; then
	PROCS=2
else
	PROCS=$1
fi

TMPDIR=/tmp/`hostname -s`-$LOGNAME
MPIS="mpich2 openmpi"

DISPLAY=1

echo "tmpdir is $TMPDIR"

. /usr/local/Modules/3.2.10/init/bash
module unload $MPIS
. ~/.bash_profile


for i in $MPIS; do
	if [ "$i" == "openmpi" ]; then
		HOSTCMD="--hostfile"
		HOSTFILE="machines-openmpi"
	elif [ "$i" == "mpich2" ]; then
		HOSTCMD="-machinefile"
		HOSTFILE="machines-mpich2"
	else
		HOSTCMD=""
	fi

	echo "Testing $i"

	module unload $MPIS
	module load $i

	bccd-snarfhosts


	cd ~/GalaxSee
	echo "GalaxSee"
	echo "  Building..."
	make clean &> /dev/null && \
	make &> /dev/null && \
	echo "  Syncing..." && \
    echo $TMPDIR && \
	bccd-syncdir --ni . ~/$HOSTFILE && \
    echo $TMPDIR && \
    ls /tmp/node000-bccd/GalaxSee.cxx-mpi && \
	echo "  Running..." && \
	runmpi $HOSTCMD ~/$HOSTFILE -np $PROCS $TMPDIR/GalaxSee.cxx-mpi 113 100 1000 $DISPLAY


	cd ~/Life
	echo "Life"
	echo "  Building..."
	make clean &> /dev/null && \
	make &> /dev/null && \
	echo "  Syncing..." && \
	bccd-syncdir --ni . ~/$HOSTFILE &> /dev/null && \
    ls /tmp/node000-bccd/Life.c-mpi && \
	echo "  Running..." && \
	runmpi $HOSTCMD ~/$HOSTFILE -np $PROCS $TMPDIR/Life.c-mpi 50 50 100 $DISPLAY


	cd ~/Parameter-space
	echo "Param-Space"
	echo "  Building..."
	make clean &> /dev/null && \
	make &> /dev/null && \
	echo "  Syncing..." && \
	bccd-syncdir --ni . ~/$HOSTFILE  &> /dev/null && \
    ls /tmp/node000-bccd/Param_Space.cxx-mpi  && \
	echo "  Running..." && \
	runmpi $HOSTCMD ~/$HOSTFILE -np $PROCS $TMPDIR/Param_Space.cxx-mpi 100 100 100 $DISPLAY

	cd ~/Pandemic
	echo "Pandemic"
	echo "  Building..."
	make clean &> /dev/null && \
	make &> /dev/null && \
	echo "  Syncing..." && \
	bccd-syncdir --ni . ~/$HOSTFILE &> /dev/null && \
    ls /tmp/node000-bccd/Pandemic.c-mpi && \
	echo "  Running..." && \
	runmpi $HOSTCMD ~/$HOSTFILE -np $PROCS $TMPDIR/Pandemic.c-mpi -t 10 $DISPLAY

    cd ~/Sieve
    echo "Sieve"
    echo "  Building..."
    make clean &> /dev/null && \
    make &> /dev/null && \
    echo "  Syncing..." && \
    bccd-syncdir --ni . ~/$HOSTFILE &> /dev/null && \
    ls /tmp/node000-bccd/Sieve.c-mpi && \
    echo "  Running..." && \
    runmpi $HOSTCMD ~/$HOSTFILE -np $PROCS $TMPDIR/Sieve.c-mpi

    cd ~/"Tree-sort"
    echo "Tree-sort"
    echo "  Building..."
    make clean &> /dev/null && \
    make &> /dev/null && \
    echo "  Syncing..." && \
    bccd-syncdir --ni . ~/$HOSTFILE &> /dev/null && \
    ls /tmp/node000-bccd/Sort.c-mpi && \
    echo "  Running..." && \
    runmpi $HOSTCMD ~/$HOSTFILE -np $PROCS $TMPDIR/Sort.c-mpi

done


echo "Done"
