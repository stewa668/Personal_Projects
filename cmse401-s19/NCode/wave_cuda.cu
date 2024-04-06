
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#define CUDA_CALL(x) {cudaError_t cuda_error__ = (x); if (cuda_error__) printf("CUDA error: " #x " returned \"%s\"\n", cudaGetErrorString(cuda_error__));}


__global__ void accel_update(double* d_dvdt, double* d_y, int nx, double dx2inv)
{
        int i = blockDim.x * blockIdx.x + threadIdx.x;
        if (i > 0 && i < nx-1)
            d_dvdt[i]=(d_y[i+1]+d_y[i-1]-2.0*d_y[i])*(dx2inv);
        else
            d_dvdt[i] = 0;
}

__global__ void pos_update(double * d_dvdt, double * d_y, double * d_v, double dt)
{
        int i = blockDim.x * blockIdx.x + threadIdx.x;
        d_v[i] = d_v[i] + dt*d_dvdt[i];
        d_y[i]  = d_y[i] + dt*d_v[i];
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

    double *d_x, *d_y, *d_v, *d_dvdt;
    CUDA_CALL(cudaMalloc((void **)&d_x,nx*sizeof(double)));
    CUDA_CALL(cudaMalloc((void **)&d_y,nx*sizeof(double)));
    CUDA_CALL(cudaMalloc((void **)&d_v,nx*sizeof(double)));
    CUDA_CALL(cudaMalloc((void **)&d_dvdt,nx*sizeof(double)));

    max=10.0;
    min=0.0;
    dx = (max-min)/(double)(nx-1);
    x[0] = min;
    for(i=1;i<nx-1;i++) {
        x[i] = min+(double)i*dx;
    }

    x[nx-1] = max;
    tmax=10.0;
    dt= (tmax-0.0)/(double)(nt-1);

    for (i=0;i<nx;i++)  {
        y[i] = exp(-(x[i]-5.0)*(x[i]-5.0));
        v[i] = 0.0;
        dvdt[i] = 0.0;
    }

   CUDA_CALL(cudaMemcpy(d_x,x,nx*sizeof(double),cudaMemcpyHostToDevice));
   CUDA_CALL(cudaMemcpy(d_y,y,nx*sizeof(double),cudaMemcpyHostToDevice));
   CUDA_CALL(cudaMemcpy(d_v,v,nx*sizeof(double),cudaMemcpyHostToDevice));
   CUDA_CALL(cudaMemcpy(d_dvdt,dvdt,nx*sizeof(double),cudaMemcpyHostToDevice));

   dx2inv=1.0/(dx*dx);
   int block_size=1024;
   int block_no = nx/block_size;
   dim3 dimBlock(block_size,1,1);
   dim3 dimGrid(block_no,1,1);

    for(it=0;it<nt-1;it++) {
        accel_update<<<dimGrid, dimBlock>>>(d_dvdt, d_y, nx, dx2inv);
        pos_update<<<dimGrid, dimBlock>>>(d_dvdt, d_y, d_v, dt);
    }

   CUDA_CALL(cudaMemcpy(x,d_x,nx*sizeof(double),cudaMemcpyDeviceToHost));
   CUDA_CALL(cudaMemcpy(y,d_y,nx*sizeof(double),cudaMemcpyDeviceToHost));

    for(i=nx/2-10; i<nx/2+10; i++) {
        printf("%g %g\n",x[i],y[i]);
    }

    return 0;
}
