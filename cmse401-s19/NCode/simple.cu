
#include "cuda.h"
#include <iostream>
#define CUDA_CALL(x) {cudaError_t cuda_error__ = (x); if (cuda_error__) std::cout << "CUDA error: " << #x << " returned " << cudaGetErrorString(cuda_error__) << std::endl;}


__global__ void theKernel(float * our_array)
{
    //This is array flattening, (Array Width * Y Index + X Index)
    int index = (gridDim.x * blockDim.x) * \
              (blockIdx.y * blockDim.y + threadIdx.y) + \
              (blockIdx.x * blockDim.x + threadIdx.x);
    our_array[index] = (float) index;
}


void printGrid(float an_array[16][16])
{
    for (int i = 0; i < 16; i++)
    {
        for (int j = 0; j < 16; j++)
        {
            std::cout << an_array[i][j];
            std::cout << " ";
        }
        std::cout << std::endl;
    }
}


int main()
{
    float our_array[16][16];

    for (int i = 0; i < 16; i++)
    {
        for (int j = 0; j < 16; j++)
        {
            our_array[i][j] = i;
        }
    }

    //STEP 1: ALLOCATE
    float * our_array_d;
    int size = sizeof(float)*256;
    CUDA_CALL(cudaMalloc((void **) &our_array_d, size));

    //STEP 2: TRANSFER
    CUDA_CALL(cudaMemcpy(our_array_d, our_array, size, cudaMemcpyHostToDevice));

    //STEP 3: SET UP
    dim3 blockSize(8,8,1);
    dim3 gridSize(2,2,1);

    //STEP 4: RUN
    theKernel <<<gridSize, blockSize>>> (our_array_d);

    //STEP 5: TRANSFER
    printGrid(our_array);
    CUDA_CALL(cudaMemcpy(our_array, our_array_d, size, cudaMemcpyDeviceToHost));
    std::cout << "--------------------" << std::endl;
    printGrid(our_array);
}
