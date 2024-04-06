#
#  Simple Makefile to compile and link trapezoid-mpi.cpp into an
#  executible trapezoid-mpi using the mpicxx compiler
#
#  David John
#  November 2013

CCC = mpicxx
RM = /bin/rm -f

#  link into an executible
trapezoid-mpi:	trapezoid-mpi.o Makefile
				$(CCC) -o trapezoid-mpi trapezoid-mpi.o

#  compile 
trapezoid-mpi.o:	trapezoid-mpi.cpp Makefile
				$(CCC) -c trapezoid-mpi.cpp

#  clean up 
clean:
				$(RM) *.o
				$(RM) *~

#  really clean up
cleanall:		clean
				$(RM) trapezoid-mpi

