# Parallelization:  Sieve of Eratosthenes
# By Aaron Weeden, Shodor Education Foundation, Inc.
# January 2012
#
# Makefile
#
# Usage:  On the command line, enter 'make' followed by one of the options below
# e.g. $ make serial
PROGRAM	= Sieve
all: serial c-openmp c-mpi c-hybrid

serial:
	icc -o $(PROGRAM).serial sieve.serial.c 
c-openmp:
	icc -openmp -o $(PROGRAM).c-openmp sieve.openmp.c 
c-mpi:
	mpicc -o $(PROGRAM).c-mpi sieve.mpi.c 
c-hybrid:
	mpicc -openmp -o $(PROGRAM).c-hybrid sieve.hybrid.c 

clean:
	rm -rf $(PROGRAM).serial $(PROGRAM).c-openmp $(PROGRAM).c-mpi $(PROGRAM).c-hybrid
