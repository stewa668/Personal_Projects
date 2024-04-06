LIBS = -lmagma -lgfortran -lmkl_gf_lp64 -lmkl_sequential -lmkl_core -lcublas -lcudart -lstdc++ -lm 

NVOPTS = -DHAVE_CUBLAS -DADD_

OBJS = \
	testing_util.o \
	dgesv_gpu.o

all: dgesv_gpu.exe

testing_util.o: testing_util.cpp
	$(CXX) -Wall $(CPPFLAGS) $(INCLUDE) testing_util.cpp -c   $(NVOPTS) $(LIBS)

dgesv_gpu.o: dgesv_gpu.cpp
	$(CXX) $(CPPFLAGS) $(INCLUDE) dgesv_gpu.cpp -c  $(NVOPTS) $(LIBS)

dgesv_gpu.exe:  $(OBJS)
	$(CXX) $(CPPFLAGS) $(INCLUDE) $(OBJS) -o $@ $(LIBS) $(NVOPTS)

clean:
	rm -f *.exe *.o
