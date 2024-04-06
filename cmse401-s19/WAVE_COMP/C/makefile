all: serial start omp minimal cuda 

cuda: wave_cuda.cu
	nvcc -o cuda wave_cuda.cu

serial: wave_serial.c
	g++ -o serial wave_serial.c

start: wave_start.c
	g++ -fopenmp -o start wave_start.c

omp: wave_omp.c
	g++ -fopenmp -o omp wave_omp.c

minimal: wave_omp_minimal.c
	g++ -fopenmp -o minimal wave_omp_minimal.c

test: all 
	time ./serial 2>&1 > output_serial.txt
	time ./cuda 2>&1 > output_cuda.txt
	time ./start 2>&1 > output_start.txt
	time ./minimal 2>&1 > output_minimal.txt
	time ./omp 2>&1 > output_omp.txt

check: output_serial.txt output_start.txt output_minimal.txt output_omp.txt
	./test_diff.sh output_serial.txt output_start.txt
	./test_diff.sh output_serial.txt output_minimal.txt
	./test_diff.sh output_serial.txt output_omp.txt
	./test_diff.sh output_serial.txt output_cuda.txt
 
clean: 
	rm -f serial start omp minimal cuda 
	rm -f output*.txt 
