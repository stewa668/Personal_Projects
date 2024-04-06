CUDAFILES:= $(wildcard *.cu)
CUDAOUTFILES:= $(patsubst %.cu,%.exe,$(CUDAFILES))

CFILES:= $(wildcard *.c)
COUTFILES:= $(patsubst %.c,%.exe,$(CFILES))

all : $(CUDAOUTFILES)  

%.exe : %.cu
	nvcc -o $@  $< 

%.exe : %.c
	g++ -lm -lOpenCL -fopenmp -o $@  $< 

clean :
	rm -f *.exe 
