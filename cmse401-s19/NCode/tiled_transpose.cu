
#include <iostream>
#include <cuda.h>
#include <chrono>
#define CUDA_CALL(x) {cudaError_t cuda_error__ = (x); if (cuda_error__) { fprintf(stderr, "CUDA error: " #x " returned \"%s\"\n", cudaGetErrorString(cuda_error__)); fflush(stderr); exit(cuda_error__); } }

using namespace std;

const int BLOCKDIM = 32; 

__global__ void transpose(const double *in_d, double * out_d, int row, int col)
{
   int x = blockIdx.x * blockDim.x + threadIdx.x;
   int y = blockIdx.y * blockDim.y + threadIdx.y;
   if (x < col && y < row) 
       out_d[y+col*x] = in_d[x+row*y];
}

__global__ void tiled_transpose(const double *in_d, double * out_d, int row, int col)
{
   int x = blockIdx.x * BLOCKDIM + threadIdx.x;
   int y = blockIdx.y * BLOCKDIM + threadIdx.y;
    
   int x2 = blockIdx.y * BLOCKDIM + threadIdx.x;
   int y2 = blockIdx.x * BLOCKDIM + threadIdx.y;
    
   __shared__ double in_local[BLOCKDIM][BLOCKDIM];
   __shared__ double out_local[BLOCKDIM][BLOCKDIM];

   if (x < col && y < row) {
       in_local[threadIdx.x][threadIdx.y] = in_d[x+row*y];
       __syncthreads();

       out_local[threadIdx.y][threadIdx.x] = in_local[threadIdx.x][threadIdx.y];
       __syncthreads();

       out_d[x2+col*y2] = out_local[threadIdx.x][threadIdx.y];
   }
}

__global__ void transpose_symmetric(double *in_d, double * out_d, int row, int col)
{
   int x = blockIdx.x * blockDim.x + threadIdx.x;
   int y = blockIdx.y * blockDim.y + threadIdx.y;
   if (x < col && y < row) {
       if (x < y) { 
           double temp =  in_d[y+col*x];
           in_d[y+col*x] = in_d[x+row*y];
           in_d[x+row*y] = temp;
       }
   }
}



int main(int argc,char **argv)
{
    std::cout << "Begin\n";
   int sz_x=BLOCKDIM*300;
   int sz_y=BLOCKDIM*300;
   int nBytes = sz_x*sz_y*sizeof(double);
   int block_size = BLOCKDIM;
   double *m_h = (double *)malloc(nBytes);
   double * in_d;
   double * out_d;
   int count = 0;
   for (int i=0; i < sz_x*sz_y; i++){
       m_h[i] = count;
       count++;
   }
   std::cout << "Allocating device memory on host..\n";
   CUDA_CALL(cudaMalloc((void **)&in_d,nBytes));
   CUDA_CALL(cudaMalloc((void **)&out_d,nBytes));

   //Set up blocks
   dim3 dimBlock(block_size,block_size,1);
   dim3 dimGrid(sz_x/block_size,sz_y/block_size,1);

   std::cout << "Doing GPU Transpose\n";
   CUDA_CALL(cudaMemcpy(in_d,m_h,nBytes,cudaMemcpyHostToDevice));
    
   auto start_d = std::chrono::high_resolution_clock::now();
   
    /**********************/
   transpose<<<dimGrid,dimBlock>>>(in_d,out_d,sz_y,sz_x);
   //tiled_transpose<<<dimGrid,dimBlock>>>(in_d,out_d,sz_y,sz_x);

   cudaError_t err = cudaGetLastError();
   if (err != cudaSuccess) {
        fprintf(stderr, "\n\nError: %s\n\n", cudaGetErrorString(err)); fflush(stderr); exit(err);   
   } 
   CUDA_CALL(cudaMemcpy(m_h,out_d,nBytes,cudaMemcpyDeviceToHost));
   /************************/
    
   /**********************
   transpose_symmetric<<<dimGrid,dimBlock>>>(in_d,out_d,sz_y,sz_x);    
   cudaError_t err = cudaGetLastError();
   if (err != cudaSuccess) {
        fprintf(stderr, "\n\nError: %s\n\n", cudaGetErrorString(err)); fflush(stderr); exit(err);   
   } 
   CUDA_CALL(cudaMemcpy(m_h,in_d,nBytes,cudaMemcpyDeviceToHost));
   ************************/
    
   auto end_d = std::chrono::high_resolution_clock::now();

   std::cout << "Doing CPU Transpose\n";
   auto start_h = std::chrono::high_resolution_clock::now();
   for (int y=0; y < sz_y; y++){
        for (int x=y; x < sz_x; x++){
           double temp = m_h[x+sz_x*y];
           //std::cout << temp << " ";
           m_h[x+sz_x*y] = m_h[y+sz_y*x];
           m_h[y+sz_y*x] = temp;
       }
       //std::cout << "\n";
   }
   auto end_h = std::chrono::high_resolution_clock::now();


   //Checking errors (should be same values as start)
   count = 0;
   int errors = 0;
   for (int i=0; i < sz_x*sz_y; i++){
       if (m_h[i] != count)
           errors++;
       count++;
   }
   std::cout << errors << " Errors found in transpose\n";

    //Print Timing
   std::chrono::duration<double> time_d = end_d - start_d;
   std::cout << "Device time: " << time_d.count() << " s\n";
   std::chrono::duration<double> time_h = end_h - start_h;
   std::cout << "Host time: " << time_h.count() << " s\n";

   cudaFree(in_d);
   cudaFree(out_d);
   return 0;
}
