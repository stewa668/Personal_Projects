#include <math.h>
#include <stdlib.h>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <device_functions.h>

#include <stdio.h>

/*Define the optimized launching configurations*/
#define BLKSIZE 128



/*Original kernel_A, used to generate the reference result*/
__global__ void kernel_1(float *d_data_in, float *d_data_out, int data_size)
{
	__shared__ float s_data[BLKSIZE];
	int tid = threadIdx.x;
	int index = tid + blockIdx.x*blockDim.x;
	s_data[tid] = 0.0;
	if (index < data_size){
		s_data[tid] = d_data_in[index];
	}
	__syncthreads();
	
	for (int s = 2; s <= blockDim.x; s = s * 2){
		if ((tid%s) == 0){
			s_data[tid] += s_data[tid + s / 2];
		}
		__syncthreads();
	}

	if (tid == 0){
		d_data_out[blockIdx.x] = s_data[tid];
	}
}

__global__ void kernel_2(float *d_data_in, float *d_data_out, int data_size)
{
	__shared__ float s_data[BLKSIZE];
	int tid = threadIdx.x;
	int index = tid + blockIdx.x*blockDim.x;
	s_data[tid] = 0.0;
	if (index < data_size){
		s_data[tid] = d_data_in[index];
	}
	__syncthreads();

	for (int s = 2; s <= blockDim.x; s = s * 2){
		index = tid * s;
		if (index < blockDim.x){
			s_data[index] += s_data[index + s / 2];
		}
		__syncthreads();
	}

	if (tid == 0){
		d_data_out[blockIdx.x] = s_data[tid];
	}
}

__global__ void kernel_3(float *d_data_in, float *d_data_out, int data_size)
{
	__shared__ float s_data[BLKSIZE];
	int tid = threadIdx.x;
	int index = tid + blockIdx.x*blockDim.x;
	s_data[tid] = 0.0;
	if (index < data_size){
		s_data[tid] = d_data_in[index];
	}
	__syncthreads();

	for (int s = blockDim.x/2; s >= 1; s = s >> 1){
		if (tid<s){
			s_data[tid] += s_data[tid + s];
		}
		__syncthreads();
	}

	if (tid == 0){
		d_data_out[blockIdx.x] = s_data[tid];
	}
}

__global__ void kernel_4(float *d_data_in, float *d_data_out, int data_size)
{
	__shared__ volatile float s_data[BLKSIZE];
	int tid = threadIdx.x;
	int index = tid + blockIdx.x*blockDim.x;
	s_data[tid] = 0.0;
	if (index < data_size){
		s_data[tid] = d_data_in[index];
	}
	__syncthreads();

	for (int s = blockDim.x / 2; s >= 64; s = s >> 1){
		if (tid<s){
			s_data[tid] += s_data[tid + s];
		}
		__syncthreads();
	}

	if (tid < 32){
		s_data[tid] += s_data[tid + 32];
		s_data[tid] += s_data[tid + 16];
		s_data[tid] += s_data[tid + 8];
		s_data[tid] += s_data[tid + 4];
		s_data[tid] += s_data[tid + 2];
		s_data[tid] += s_data[tid + 1];
	}

	if (tid == 0){
		d_data_out[blockIdx.x] = s_data[tid];
	}
}

__global__ void kernel_5(float *d_data_in, float *d_data_out, int data_size)
{
	__shared__ volatile float s_data[BLKSIZE];
	int tid = threadIdx.x;
	int index = tid + blockIdx.x*blockDim.x*2;
	s_data[tid] = 0.0;
	if (index < data_size){
		s_data[tid] = d_data_in[index];
	}
	if ((index + blockDim.x) < data_size){
		s_data[tid] += d_data_in[index + blockDim.x];
	}
	__syncthreads();

	for (int s = blockDim.x / 2; s >= 64; s = s >> 1){
		if (tid<s){
			s_data[tid] += s_data[tid + s];
		}
		__syncthreads();
	}

	if (tid < 32){
		s_data[tid] += s_data[tid + 32];
		s_data[tid] += s_data[tid + 16];
		s_data[tid] += s_data[tid + 8];
		s_data[tid] += s_data[tid + 4];
		s_data[tid] += s_data[tid + 2];
		s_data[tid] += s_data[tid + 1];
	}

	if (tid == 0){
		d_data_out[blockIdx.x] = s_data[tid];
	}
}

template <int blockSize>
__global__ void kernel_6(float *d_data_in, float *d_data_out, int data_size)
{
	__shared__ volatile float s_data[blockSize];
	int tid = threadIdx.x;
	int index = tid + blockIdx.x*blockDim.x * 2;
	s_data[tid] = 0.0;
	if (index < data_size){
		s_data[tid] = d_data_in[index];
	}
	if ((index + blockDim.x) < data_size){
		s_data[tid] += d_data_in[index + blockDim.x];
	}
	__syncthreads();

	if (blockSize >= 1024){
		if (tid < 512){
			s_data[tid] += s_data[tid + 512];
		}
		__syncthreads();
	}
	if (blockSize >= 512){
		if (tid < 256){
			s_data[tid] += s_data[tid + 256];
		}
		__syncthreads();
	}
	if (blockSize >= 256){
		if (tid < 128){
			s_data[tid] += s_data[tid + 128];
		}
		__syncthreads();
	}
	if (blockSize >= 128){
		if (tid < 64){
			s_data[tid] += s_data[tid + 64];
		}
		__syncthreads();
	}

	if (tid < 32){
		s_data[tid] += s_data[tid + 32];
		s_data[tid] += s_data[tid + 16];
		s_data[tid] += s_data[tid + 8];
		s_data[tid] += s_data[tid + 4];
		s_data[tid] += s_data[tid + 2];
		s_data[tid] += s_data[tid + 1];
	}

	if (tid == 0){
		d_data_out[blockIdx.x] = s_data[tid];
	}
}

int launch_kernel(void(*kernel)(float*, float*, int), float *d_data_in, float **d_data_out, int data_size, int* out_size)
{
	int grid_size = data_size / (BLKSIZE * 2) + ((data_size % (BLKSIZE * 2)) == 0 ? 0 : 1);
	dim3 block(BLKSIZE);
	dim3 grid(grid_size);

	cudaMalloc((void**)d_data_out, grid_size*sizeof(float));
	
	//switch (BLKSIZE){
	//case 512:
		//kernel<512> << <grid, block >> >(d_data_in, *d_data_out, data_size);



	//}

	cudaDeviceSynchronize();

	*out_size = grid_size;
	return 0;
}

float reduction_recursive_cuda(float *d_data, int data_size, int flag){

	float *d_result;
	int out_size;
	launch_kernel(&kernel_5, d_data, &d_result, data_size, &out_size);
	if (flag){
		cudaFree(d_data);
	}

	if (out_size == 1){
		float h_result;
		cudaMemcpy(&h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost);
		return h_result;
	}
	else{
		return reduction_recursive_cuda(d_result, out_size, 1);
	}
}

float reduction_cpu(float *h_data, int data_size){

	float sum = 0.0;
	for (int i = 0; i < data_size; i++){
		sum += h_data[i];
	}
	return sum;
}




float timing_experiment_cpu(float (*func)(float*, int), float *h_data, int data_size, int nreps, float *result)
{
	float elapsed_time_ms=0.0f;
	cudaEvent_t start, stop;
	cudaEventCreate( &start );
	cudaEventCreate( &stop  );

	cudaEventRecord( start, 0 );
	for (int i = 0; i < nreps; i++){
		*result = func(h_data, data_size);
	}
	cudaEventRecord( stop, 0 );
	cudaThreadSynchronize();
	cudaEventElapsedTime( &elapsed_time_ms, start, stop );
	elapsed_time_ms /= nreps;

	cudaEventDestroy( start );
	cudaEventDestroy( stop );

	return elapsed_time_ms;
}

float timing_experiment_gpu(float(*func)(float*, int, int), float *d_data, int data_size, int nreps, float *result)
{
	float elapsed_time_ms = 0.0f;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start, 0);
	for (int i = 0; i < nreps; i++){
		*result = func(d_data, data_size,0);
	}
	cudaEventRecord(stop, 0);
	cudaThreadSynchronize();
	cudaEventElapsedTime(&elapsed_time_ms, start, stop);
	elapsed_time_ms /= nreps;

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	return elapsed_time_ms;
}



int main()
{
	int dimx = 2*8*1024;
	int dimy = 2*8*1024;
	int data_size = dimx*dimy;
	int nreps;
	int nbytes = data_size*sizeof(float);

	int seed = time(NULL);
	srand(seed);

	float *d_data = 0;
	float *h_data = 0;
	cudaMalloc( (void**)&d_data, nbytes );
	if( 0 == d_data )
	{
		printf("couldn't allocate GPU memory\n");
		return -1;
	}
	printf("allocated %.2f MB on GPU\n", nbytes/(1024.f*1024.f) );
	h_data = (float*)malloc( nbytes );
	if( 0 == h_data )
	{
		printf("couldn't allocate CPU memory\n");
		return -2;
	}
	printf("allocated %.2f MB on CPU\n", nbytes/(1024.f*1024.f) );
	for(int i=0; i<dimx*dimy; i++)
		h_data[i] = 1.f + (float)rand() / (float)RAND_MAX;
	cudaMemcpy( d_data, h_data, nbytes, cudaMemcpyHostToDevice );

	float result_cpu,result_CUDA;
	
	float elapsed_time_ms = 0.0f;
	nreps = 1;
	//elapsed_time_ms = timing_experiment_cpu(reduction_cpu, h_data, data_size, nreps, &result_cpu);
	//printf("CPU reduction:  %8.8f ms, result = %f\n", elapsed_time_ms, result_cpu);

	elapsed_time_ms = timing_experiment_gpu(reduction_recursive_cuda, d_data, data_size, nreps, &result_CUDA);
	printf("CUDA reduction:  %8.8f ms, result = %f\n", elapsed_time_ms, result_CUDA);

	printf("CUDA: %s\n", cudaGetErrorString( cudaGetLastError() ) );

	if( d_data )
		cudaFree( d_data );
	if( h_data )
		free( h_data );

	cudaThreadExit();

	return 0;
}
