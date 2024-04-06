transpose: transpose.c
	gcc -O3 -o transpose transpose.c


test: transpose
	time ./transpose 20 16 0 0 
	time ./transpose 100 16 0 0 
	time ./transpose 500 16 0 0 
	time ./transpose 1000 16 0 0 
	time ./transpose 1500 16 0 0 
	time ./transpose 2000 16 0 0 
	time ./transpose 4000 16 0 0 
	time ./transpose 20 16 1 0 
	time ./transpose 100 16 1 0 
	time ./transpose 500 16 1 0 
	time ./transpose 1000 16 1 0 
	time ./transpose 1500 16 1 0 
	time ./transpose 2000 16 1 0 
	time ./transpose 4000 16 1 0 

