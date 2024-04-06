#
# Simple Makefile to compile and link traepzoid-cuda.cu to produce
# the executable trapezoid-cuda
#
# David John
# November 2013
# @copyright 2013, Wake Forest University

CCC = nvcc
CFLAGS = -O2 
ARCH := $(shell uname -m)

ifeq ($(ARCH), i686)
	CFLAGS += --machine 32
endif

ifeq ($(ARCH), x86_64)
	CFLAGS += --machine 64
endif



trapezoid-cuda : trapezoid-cuda.o Makefile
	$(CCC) $(CFLAGS) -o trapezoid-cuda trapezoid-cuda.o

trapezoid-cuda.o: trapezoid-cuda.cu Makefile
	$(CCC) -c trapezoid-cuda.cu

clean:
	$(RM) *.o
	$(RM) *~

cleanall:  clean
	rm -f trapezoid-cuda
