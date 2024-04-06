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
# Makefile for the modules included in ~bccd on the BCCD (http://bccd.net)
# 
# Add NO_X11=1 to the make command line to omit X libraries.
# Add DEBUG=1 to add -ggdb to CFLAGS
# Add PROFILE=1 to add -pg to CFLAGS and LDFLAGS
# Add STAT_KIT=1 to enable StatKit
###############################################################################

PROGRAM = GalaxSee-v2
CSRCS = nbody.c text11.c octtree.c mem.c rand_tools.c quaternion.c sdlwindow.c\
		fcr.c pppm.c cubeinterp.c readline.c gal2gad2.c galaxsee.c
INCS = cubeinterp.h nbody.h pppm_structs.h sdlwindow.h fcr.h octtree.h\
       quaternion.h text11.h gal2gad2.h octtree_structs.h rand_tools.h\
       mem.h pppm.h readline.h
CLEAN = $(PROGRAM).c-mpi
		
ifdef STAT_KIT
	IGNORE := $(shell cd ../StatKit/petakit; make)
endif		

all: c-mpi

c-mpi: $(CSRCS) $(INCS) $(PROGRAM).c-mpi
$(PROGRAM).c-mpi: $(CSRCS) $(INCS)

vidclean:;	rm -rf out*.png out*.pov
povfiles:;	for f in *.dump; do make_pov.perl $$f >& /dev/null; done
vidpov:;	for f in *.pov; do povray -H600 -W800 -D $$f >& /dev/null; done
vidpovpbs:;	for f in *.pov; do qsub -v FILE=$$f frames.qsub ; done
anim:;		convert -delay 1 -loop 0 -quality 100 out*png anim.gif

FFTW :=	1
#NETCDF	:=	1
#GADGET2 :=	1
#LIBGD :=	1

include	../Makefile
