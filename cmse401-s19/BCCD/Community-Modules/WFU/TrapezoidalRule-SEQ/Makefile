#
#  Simple Makefile to compile and link trapezoid.cpp into
#  the executible trapezoid-seq
#
#  David John
#  November 2013
#

CCC = g++
RM = /bin/rm -f

#  Link pieces together to produce trapezoid-seq
trapezoid-seq:	trapezoid.o Makefile
				$(CCC) -o trapezoid-seq trapezoid.o

#  Compile trapezoid.cpp to get trapezoid.o
trapezoid.o:	trapezoid.cpp Makefile
				$(CCC) -c trapezoid.cpp

#  Clean up directory
clean:
				${RM} *.o
				${RM} *.~

#  Really clean this up
cleanall:		clean
				${RM} trapezoid-seq

