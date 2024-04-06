all: process 

main_process.o: main_process.c
	gcc -O3 -Ofast -c main_process_original.c -fopenmp
	gcc -O3 -Ofast -c main_process_opt_serial.c -fopenmp
	gcc -O3 -Ofast -c main_process.c -fopenmp

png_util.o: png_util.c
	gcc -O3 -Ofast -l lpng16 -c png_util.c -fopenmp

process: main_process.o png_util.o
	gcc -O3 -Ofast -o process_original -lm -l png16 main_process_original.o png_util.o -fopenmp
	gcc -O3 -Ofast -o process_opt_serial -lm -l png16 main_process_opt_serial.o png_util.o -fopenmp
	gcc -O3 -Ofast -o process -lm -l png16 main_process.o png_util.o -fopenmp

test: process 
	./process cube.png test.png

clean:
	rm *.o
	rm process 
