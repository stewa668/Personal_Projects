#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#define CUDA_CALL(x) {cudaError_t cuda_error__ = (x); if (cuda_error__) { fprintf(stderr, "CUDA error: " #x " returned \"%s\"\n", cudaGetErrorString(cuda_error__)); fflush(stderr); exit(cuda_error__); } }

const int BLOCKDIM=1024;

__global__ void accel_update(double* d_dvdt, double* d_y, int nx, double dx2inv)
{
	int i = (blockDim.x * blockIdx.x + threadIdx.x)+1;
	if (i < nx-1)
        d_dvdt[i] = (d_y[i+1]+d_y[i-1]-2.0*d_y[i])*dx2inv;
}

__global__ void pos_update(double * d_dvdt, double * d_y, double * d_v, int nx, double dt)
{
	int i = (blockDim.x * blockIdx.x + threadIdx.x)+1;
    if (i < nx-1) {
        d_v[i] = d_v[i] + dt*d_dvdt[i];
        d_y[i] = d_y[i] + dt*d_v[i];
    }  
}

const int LOCALDIM=BLOCKDIM+2;
__global__ void tile_accel_update(double* d_dvdt, double* d_y, int nx, double dx2inv)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
    int mx = blockDim.x * (blockIdx.x + 1) - 1;
    
    //Allocate local memory
    __shared__ double y_local[LOCALDIM];  
    
    //Copy to local index
    int local_idx = threadIdx.x+1; 
    y_local[local_idx] = d_y[i]; 
    
    //fill in edges
    if(threadIdx.x == 0) {
        y_local[0] = d_y[i-1];  
        y_local[LOCALDIM-1] = d_y[mx]; 
    }    
    
    //Check for edge case
    if(i == 0 || i > nx-2)  
        d_dvdt[i] = 0;
    else {
        d_dvdt[i]=(y_local[local_idx+1]+y_local[local_idx-1]-2.0*y_local[local_idx])*(dx2inv); 
        d_dvdt[i] = (d_y[i+1]+d_y[i-1]-2.0*d_y[i])*dx2inv;
    }
}

__global__ void all(double* d_dvdt, double* d_y,  double * d_v, int nx, double dx2inv, double dt, int nt)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
    int mx = blockDim.x * (blockIdx.x + 1) - 1;
    
    //Allocate local memory
    __shared__ double y_local[LOCALDIM];  
    
    //Copy to local index
    int local_idx = threadIdx.x+1;  
    y_local[local_idx] = d_y[i]; 
    
    //fill in edges
    if(threadIdx.x == 0) {
        y_local[0] = d_y[i-1];  
        y_local[LOCALDIM-1] = d_y[mx]; 
    }    
    
    //Check for edge case
    if(i == 0 || i > nx-2)  
        d_dvdt[i] = 0;
    else {
        for(int it=0;it<nt;it++) {
            //fill in edges
            d_dvdt[i]=(y_local[local_idx+1]+y_local[local_idx-1]-2.0*y_local[local_idx])*(dx2inv); 

            d_v[i] = d_v[i] + dt*d_dvdt[i];
            y_local[local_idx] = y_local[local_idx] + dt*d_v[i];
            
        }
    }
    d_y[i] = y_local[local_idx];
}

    
int main(int argc, char ** argv) {
    int nx = 5000;
    int nt = 1000000;
    int i,it;
    double x[nx];
    double y[nx];
    double v[nx];
    double dvdt[nx];
    double dt;
    double dx;
    double max,min;
    double dx2inv;
    double tmax;

    max=10.0;
    min=0.0;
    dx = (max-min)/(double)(nx);
    x[0] = min;
    for(i=1;i<nx;i++) {
        x[i] = min+(double)i*dx;
    }
    x[nx-1] = max;
    tmax=10.0;
    dt= (tmax-0.0)/(double)(nt);

    for (i=0;i<nx;i++)  {
        y[i] = exp(-(x[i]-5.0)*(x[i]-5.0));
        v[i] = 0.0;
        dvdt[i] = 0.0;
    }
    
    dx2inv=1.0/(dx*dx);

    double *d_x, *d_y, *d_v, *d_dvdt;
    cudaError_t err; 
    int nBytes = nx*sizeof(double);
    CUDA_CALL(cudaMalloc((void **)&d_x,nBytes));
    CUDA_CALL(cudaMalloc((void **)&d_y,nBytes));
    CUDA_CALL(cudaMalloc((void **)&d_v,nBytes));
    CUDA_CALL(cudaMalloc((void **)&d_dvdt,nBytes));
    
   fprintf(stderr, "dt = %f, dx = %f\n", dt,dx);
   CUDA_CALL(cudaMemcpy(d_y,y,nBytes,cudaMemcpyHostToDevice));
   CUDA_CALL(cudaMemcpy(d_v,v,nBytes,cudaMemcpyHostToDevice));
   CUDA_CALL(cudaMemcpy(d_dvdt,dvdt,nBytes,cudaMemcpyHostToDevice));
    
   dx2inv=1.0/(dx*dx);
    
   int block_size=BLOCKDIM;
   int block_no = (nx-2)/block_size;
   dim3 dimBlock(block_size,1,1);
   dim3 dimGrid(block_no,1,1);

    /******************************/
    for(it=0;it<200000;it++) {
        
        accel_update<<<dimGrid, dimBlock>>>(d_dvdt, d_y, nx, dx2inv);
        //tile_accel_update<<<dimGrid, dimBlock>>>(d_dvdt, d_y, nx, dx2inv);
        
        err = cudaGetLastError();
        if (err != cudaSuccess) {
             fprintf(stderr, "\n\nError: %s\n\n", cudaGetErrorString(err)); fflush(stderr); exit(err);   
        }
        
        pos_update<<<dimGrid, dimBlock>>>(d_dvdt, d_y, d_v,nx, dt);
        
         err = cudaGetLastError();
        if (err != cudaSuccess) {
             fprintf(stderr, "\n\nError: %s\n\n", cudaGetErrorString(err)); fflush(stderr); exit(err);   
        }
    }
    /******************************/
        
    CUDA_CALL(cudaMemcpy(y,d_y,nBytes,cudaMemcpyDeviceToHost));

    //printf("x, y\n");
    for(i=0; i<nx; i++) {
        printf("%g, %g\n",x[i],y[i]);
    }
    
    cudaFree(d_dvdt);
    cudaFree(d_y);
    cudaFree(d_v);

    return 0;
}
