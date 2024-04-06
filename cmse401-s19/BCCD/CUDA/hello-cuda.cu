/*
  * $Id: hello-cuda.cu,v 1.3 2012/05/01 13:53:22 charliep Exp $
  * 
  * This file is part of BCCD, an open-source live CD for computational science
  * education.
  * 
  * Copyright (C) 2010 Andrew Fitz Gibbon, Paul Gray, Kevin Hunter, Dave 
  *   Joiner, Sam Leeman-Munk, Tom Murphy, Charlie Peck, Skylar Thompson, 
  *   & Aaron Weeden 
  * 
  * This program is free software: you can redistribute it and/or modify
  * it under the terms of the GNU General Public License as published by
  * the Free Software Foundation, either version 3 of the License, or
  * (at your option) any later version.
  * 
  * This program is distributed in the hope that it will be useful,
  * but WITHOUT ANY WARRANTY; without even the implied warranty of
  * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  * GNU General Public License for more details.
  * 
  * You should have received a copy of the GNU General Public License
  * along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

/*
  * CUDA Hello World Program
  *
  * Usage: hello-cuda <number thread blocks> <number threads per block>
  *
  * charliep	09-April-2011	First pass, based on the example by Alan Kaminsky
  * charliep	01-July-2011	Improved error handling.
*/

#include <stdlib.h>
#include <stdio.h>

/*
  * Kernel function, this runs on the GPGPU chip. This thread's element of barray 
  * is set to this thread's block index. This thread's element of tarray is set 
  * to this thread's thread index within the block.  Note the use of the builtin
  * variables blockDim, blockIdx and threadIdx.
*/
__global__ void hello(int* barray, int* tarray) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	barray[i] = blockIdx.x;
	tarray[i] = threadIdx.x;
}

void usage() {
	fprintf(stderr, "Usage: hello-cuda <number thread blocks> <number threads per block>\n");
	exit(1);
}

int main(int argc, char** argv) {
	int numThreadBlocks, numThreadsPerBlock, totalNumThreads, size, i;
	int *cpuBlockArray, *cpuThreadArray, *gpuBlockArray, *gpuThreadArray;
	cudaError_t status = (cudaError_t)0; 

	if (argc != 3) 
		usage();
	if (sscanf(argv[1], "%d", &numThreadBlocks) != 1) 
		usage();
	if (sscanf(argv[2], "%d", &numThreadsPerBlock) != 1) 
		usage();
	
	totalNumThreads = numThreadBlocks * numThreadsPerBlock; 

	/* Allocate CPU memory. */ 
	size = totalNumThreads * sizeof(int);
	
	if (!(cpuBlockArray = (int*) malloc(size))) {
		fprintf(stderr, "malloc() FAILED (block)\n"); 
		exit(0);
	}
	
	if (!(cpuThreadArray = (int*) malloc(size))) {
		fprintf(stderr, "malloc() FAILED (thread)\n"); 
		exit(0);
	}
	
	/* Allocate GPGPU memory. */ 
	if ((status = cudaMalloc ((void**) &gpuBlockArray, size)) != cudaSuccess) {
		printf("cudaMalloc() FAILED (block), status = %d (%s)\n", status, cudaGetErrorString(status));
		exit(1); 
	}

	if ((status = cudaMalloc ((void**) &gpuThreadArray, size)) != cudaSuccess) {
		printf("cudaMalloc() FAILED (thread), status = %d (%s)\n", status, cudaGetErrorString(status));
		exit(1); 
	}
	
	/* Call the kernel function to run on the GPGPU chip. */ 
	hello <<<numThreadBlocks, numThreadsPerBlock>>> 
	  (gpuBlockArray, gpuThreadArray);
	
	/* Copy the result arrays from the GPU's memory to the CPU's memory. */ 
	cudaMemcpy(cpuBlockArray, gpuBlockArray, size, cudaMemcpyDeviceToHost);
	cudaMemcpy(cpuThreadArray, gpuThreadArray, size, cudaMemcpyDeviceToHost);
	
	/* Display the results. */ 
	for (i = 0; i < totalNumThreads; ++i) {
		printf("%d\t%d\n", cpuBlockArray[i], cpuThreadArray[i]);
	}
	
	printf("Total number of hellos: %d\n", totalNumThreads); 
	
	/* Free CPU and GPU memory. */
	free(cpuBlockArray);
	free(cpuThreadArray);
	cudaFree(gpuBlockArray);
	cudaFree(gpuThreadArray);
	
	exit(0); 
}
