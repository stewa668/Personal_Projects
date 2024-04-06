# Makefile for Parallelization: Binary Tree Traversal
# Aaron Weeden and Patrick Royal, Shodor Education Foundation, Inc.
# April 2012
#
# Usage: make [clean|all|sort|sort-parallel]

CC=icc
MPICC=mpicc

MPICFLAGS=$(CFLAGS) -DPARALLEL

PROGRAM	=	Sort

all:	serial	c-mpi

serial: tree.h sort.c
	$(CC) -o $(PROGRAM).serial sort.c 

c-mpi: tree.h sort.c
	$(MPICC) $(MPICFLAGS) -o $(PROGRAM).c-mpi sort.c 

clean:
	rm -f $(PROGRAM).serial $(PROGRAM).c-mpi

