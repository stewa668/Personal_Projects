
//Example modified from: https://gist.github.com/vo/3899348
//Timing code from: https://www.pluralsight.com/blog/software-development/how-to-measure-execution-time-intervals-in-c--

#include <iostream>
#include <cuda.h>
#include <chrono>
#define CUDA_CALL(x) {cudaError_t cuda_error__ = (x); if (cuda_error__) { fprintf(stderr, "CUDA error: " #x " returned \"%s\"\n", cudaGetErrorString(cuda_error__)); fflush(stderr); exit(cuda_error__); } }

__global__ void vecAdd(int *a_d,int *b_d,int *c_d,int N)
{
   int i = blockDim.x * blockIdx.x + threadIdx.x;
   if (i < N)
       c_d[i] = a_d[i] + b_d[i]; 
}

void vecAdd_h(int *A1,int *B1, int *C1, int N)
{
   for(int i=0;i<N;i++)
      C1[i] = A1[i] + B1[i];
}

int main(int argc,char **argv)
{
   int n=10000000;
   int nBytes = n*sizeof(int);
   int *a,*b,*c,*c2;
   int *a_d,*b_d,*c_d;

   int num_threads = 1024;
   int num_blocks = n/num_threads+1;
   dim3 numThreads(num_threads,1,1);
   dim3 numBlocks(num_blocks,1,1); 
    
   //Check device
   struct cudaDeviceProp properties;
   cudaGetDeviceProperties(&properties, 0);
   printf("using %d multiprocessors\n",properties.multiProcessorCount);
   printf("max threads per processor: %d \n\n",properties.maxThreadsPerMultiProcessor);
    
    
   printf("nBytes=%d num_threads=%d, num_blocks=%d\n",nBytes,num_threads,num_blocks);

   if (!(a = (int*) malloc(nBytes))) {
        fprintf(stderr, "malloc() FAILED (thread)\n");
        exit(0);
    }

   if (!(b = (int*) malloc(nBytes))) {
        fprintf(stderr, "malloc() FAILED (thread)\n");
        exit(0);
    }

   if (!(c = (int*) malloc(nBytes))) {
        fprintf(stderr, "malloc() FAILED (thread)\n");
        exit(0);
    }

   if (!(c2 = (int*) malloc(nBytes))) {
        fprintf(stderr, "malloc() FAILED (thread)\n");
        exit(0);
    }
    
   for(int i=0;i<n;i++)
      a[i]=i,b[i]=i;
    
   printf("Allocating device memory on host..\n");
   CUDA_CALL(cudaMalloc((void **)&a_d,nBytes));
   CUDA_CALL(cudaMalloc((void **)&b_d,nBytes));
   CUDA_CALL(cudaMalloc((void **)&c_d,nBytes));
    
   auto start_d = std::chrono::high_resolution_clock::now();

   printf("Copying to device..\n");
   CUDA_CALL(cudaMemcpy(a_d,a,nBytes,cudaMemcpyHostToDevice));
   CUDA_CALL(cudaMemcpy(b_d,b,nBytes,cudaMemcpyHostToDevice));
   
   printf("Doing GPU Vector add\n");
   vecAdd<<<numBlocks, numThreads>>>(a_d,b_d,c_d,n);

   cudaError_t err = cudaGetLastError();
   if (err != cudaSuccess) {
        fprintf(stderr, "\n\nError: %s\n\n", cudaGetErrorString(err)); fflush(stderr); exit(err);   
   }
    
   printf("Copying results to host..\n");   
   CUDA_CALL(cudaMemcpy(c,c_d,nBytes,cudaMemcpyDeviceToHost));
   
   auto end_d = std::chrono::high_resolution_clock::now();
   
   auto start_h = std::chrono::high_resolution_clock::now();
   printf("Doing CPU Vector add\n");
   vecAdd_h(a,b,c2,n);
   auto end_h = std::chrono::high_resolution_clock::now();
    
   //Test results
   int error = 0;
   for(int i=0;i<n;i++) {
      error += abs(c[i]-c2[i]);
      if (error)
          printf("%i, %d, %d\n", i, c[i], c2[i]);
   }

   //Print Timing
   std::chrono::duration<double> time_d = end_d - start_d;
   std::chrono::duration<double> time_h = end_h - start_h;
   printf("vectorsize=%d\n",n);
   printf("difference_error=%d\n",error);
   printf("Device time: %f s\n ", time_d.count());
   printf("Host time: %f s\n", time_h.count()); 
    
   cudaFree(a_d);
   cudaFree(b_d);
   cudaFree(c_d);
   return 0;
}
