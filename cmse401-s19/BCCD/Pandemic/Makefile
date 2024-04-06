# DESCRIPTION:  Makefile for serial/parallel codes in C
# AUTHOR:       Aaron Weeden, Shodor Education Foundation, Inc.
# DATE:         Original for Area Under A Curve module, September, 2011.
#		Modified for Infectious Disease module, November, 2011.

# Code prefix
PROGRAM_PREFIX=Pandemic

# Compilers and flags
CC=icc
MPICC=mpicc
#CFLAGS+=-DTEXT_DISPLAY # Uncomment to show text display
#CFLAGS+=-DX_DISPLAY -L/usr/X11R6/lib -lX11 # Uncomment to show X display
CFLAGS+=-DSHOW_RESULTS # Uncomment to make the program print its results

# Source files
SRCS=$(PROGRAM_PREFIX).c

# OpenMP
OPENMP_FLAGS=-fopenmp -DOMP

# MPI
MPI_FLAGS=-D_MPI

# Hybrid MPI/OpenMP
HYBRID_FLAGS=$(MPI_FLAGS) $(OPENMP_FLAGS)

# Make targets
all: serial c-openmp c-mpi c-hybrid

serial: $(PROGRAM_PREFIX).serial

c-openmp: $(PROGRAM_PREFIX).c-openmp

c-mpi: $(PROGRAM_PREFIX).c-mpi

c-hybrid: $(PROGRAM_PREFIX).c-hybrid

clean:
	rm -f $(PROGRAM_PREFIX).serial $(PROGRAM_PREFIX).c-openmp $(PROGRAM_PREFIX).c-mpi $(PROGRAM_PREFIX).c-hybrid

# Make rules
$(PROGRAM_PREFIX).serial: $(SRCS)
	$(CC) $(CFLAGS) $(SRCS) -o $(PROGRAM_PREFIX).serial

$(PROGRAM_PREFIX).c-openmp: $(SRCS)
	$(CC) $(CFLAGS) $(OPENMP_FLAGS) $(SRCS) -o $(PROGRAM_PREFIX).c-openmp

$(PROGRAM_PREFIX).c-mpi: $(SRCS)
	$(MPICC) $(CFLAGS) $(MPI_FLAGS) $(SRCS) -o $(PROGRAM_PREFIX).c-mpi

$(PROGRAM_PREFIX).c-hybrid: $(SRCS)
	$(MPICC) $(CFLAGS) $(HYBRID_FLAGS) $(SRCS) -o $(PROGRAM_PREFIX).c-hybrid

