all : pkit.o

pkit.o :
	cc -c pkit.c -o pkit-c.o
	g++ -c pkit.c -o pkit-cxx.o

clean:
	/bin/rm -f pkit-c.o pkit-cxx.o

