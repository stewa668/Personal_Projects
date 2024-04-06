%%writefile transpose.cu

#include <iostream>
#include <cuda.h>
#include <chrono>

using namespace std;

__global__ void transpose(double *in_d, double * out_d, int row, int col)
{
   int x = blockIdx.x * blockDim.x + threadIdx.x;
   int y = blockIdx.y * blockDim.y + threadIdx.y;
   out_d[y+col*x] = in_d[x+row*y];
}

int main(int argc,char **argv)
{
   int sz_x=32*300;
   int sz_y=32*300;
   int nBytes = sz_x*sz_y*sizeof(double);
   int block_size;
   double *m_h = (double *)malloc(nBytes);
   double * in_d;
   double * out_d;
   int count = 0;
   for (int i=0; i < sz_x*sz_y; i++){
       m_h[i] = count;
       count++;
   }
   std::cout << "Allocating device memory on host..\n";
   auto start_d = std::chrono::high_resolution_clock::now();
   cudaMalloc((void **)&in_d,nBytes);
   cudaMalloc((void **)&out_d,nBytes);

  //Set up blocks
   block_size=32;
   dim3 dimBlock(block_size,block_size,1);
   dim3 dimGrid(sz_x/block_size,sz_y/block_size,1);

   std::cout << "Doing GPU Transpose\n";
   cudaMemcpy(in_d,m_h,nBytes,cudaMemcpyHostToDevice);
   transpose<<<dimGrid,dimBlock>>>(in_d,out_d,sz_y,sz_x);
   cudaMemcpy(m_h,out_d,nBytes,cudaMemcpyDeviceToHost);
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
