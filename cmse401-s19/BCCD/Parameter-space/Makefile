# $\Id$ 
#
# This file is part of BCCD, an open-source live CD for computational science
# education.
# 
# Copyright (C) 2010 Andrew Fitz Gibbon, Paul Gray, Kevin Hunter, Dave Joiner, 
#   Sam Leeman-Munk, Tom Murphy, Charlie Peck, Skylar Thompson, & Aaron Weeden
 
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

###############################################################################
# Makefile for the Parameter Space as included in ~bccd on the BCCD (http://bccd.net)
#
# Add NO_X11=1 to the make command line to omit X libraries.
# Add DEBUG=1 to add -ggdb to CFLAGS
# Add PROFILE=1 to add -pg to CFLAGS and LDFLAGS
# Add STAT_KIT=1 to enable StatKit
###############################################################################

PROGRAM = Param_Space
CXXSRCS = compute.cpp do_stuff.cpp param_space.cpp
INCS = compute.h do_stuff.h param_space.h
CLEAN = $(PROGRAM).cxx-mpi

ifdef STAT_KIT
	IGNORE := $(shell cd ../StatKit/petakit; make)
endif

all: cxx-mpi

cxx-mpi: $(CXXSRCS) $(INCS) $(PROGRAM).cxx-mpi
$(PROGRAM).cxx-mpi: $(CXXSRCS) $(INCS)

include ../Makefile
