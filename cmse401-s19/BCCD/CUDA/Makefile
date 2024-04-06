# $\Id$ 
#
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

CC = nvcc
CFLAGS = -O2 
ARCH := $(shell uname -m)

ifeq ($(ARCH), i686)
	CFLAGS += --machine 32
endif

ifeq ($(ARCH), x86_64)
	CFLAGS += --machine 64
endif

ifeq ($(wildcard /etc/bccd-revision),)      # not the BCCD
  TARGETS = device-query device-query-mpi hello-cuda
  LDFLAGS = 
else                                        # the BCCD
  # The combination of MPI and CUDA and icc needs to be reconciled before 
  # device-query-mpi can be built, separate functions into CUDA and not
  TARGETS = device-query hello-cuda 
endif

all: $(TARGETS)

device-query : device-query-local.c device-query-functions.h
	$(CC) $(CFLAGS) -o $@ device-query-local.c $(LDFLAGS)

device-query-mpi : device-query-mpi.c device-query-functions.h
	$(CC) $(CFLAGS) --compiler-bindir='mpicc' -o $@ device-query-mpi.c $(LDFLAGS)

hello-cuda : hello-cuda.cu
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

clean :
	rm -f device-query hello-cuda device-query-mpi
