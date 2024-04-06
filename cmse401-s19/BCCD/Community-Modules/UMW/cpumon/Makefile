# $Id: Makefile 4299 2013-09-08 01:21:19Z skylar $

MPICXX = mpiCC
LDFLAGS = -lX11
EXPENDABLES = cpumonitor

cpumon: cpumon_v1.0.cpp
	$(MPICXX) cpumon_v1.0.cpp $(LDFLAGS) -o cpumonitor

clean:
	/bin/rm -f $(EXPENDABLES)
