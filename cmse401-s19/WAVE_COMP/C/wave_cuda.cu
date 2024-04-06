#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#define CUDA_CALL(x) {cudaError_t cuda_error__ = (x); if (cuda_error__) printf("CUDA error: " #x " returned \"%s\"\n", cudaGetErrorString(cuda_error__));}

const int BLOCKDIM=1024;
const int LOCALDIM=BLOCKDIM+2;

__global__ void accel_update(double* d_dvdt, double* d_y, int nx, double dx2inv)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x+1;
	if (i < nx)
        d_dvdt[i]= (d_y[i+1]+d_y[i-1]-2.0*d_y[i])*(dx2inv);
 	else
	    d_dvdt[i] = 0;
}

__global__ void pos_update(double * d_dvdt, double * d_y, double * d_v, double dt)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x+1;
    d_v[i] = d_v[i] + dt*d_dvdt[i];
    d_y[i] = d_y[i] + dt*d_v[i];
}

__global__ void tile_accel_update(double* d_dvdt, double* d_y, int nx, double dx2inv)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
    int mini = blockDim.x * blockIdx.x;   // 1 & 2
    int maxi = (blockDim.x * blockIdx.x + 1) - 1; // 3 & 4 & 5
    //Allocate local memory
    __shared__ double y_local[LOCALDIM];  // 6
    
    
    //Copy to local index
    int local_idx = threadIdx.x+1;  // 7 & 8
    y_local[local_idx] = d_y[i]; 
    
    //Check for edge case
    if(i == 0 || i < nx)  
        d_dvdt[i] = 0;
    else {
        //fill in edges
        if(local_idx == 1) {     // 9 & 10  
            y_local[0] = d_y[mini];  
            y_local[LOCALDIM-1] = d_y[maxi]; 
        }

        d_dvdt[i]=(y_local[local_idx+1]+y_local[local_idx-1]-2.0*y_local[local_idx])*(dx2inv); 
    }
}




__global__ void all(double* d_dvdt, double* d_y,  double * d_v, int nx, double dx2inv, double dt, int nt)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
    int mini = blockDim.x * blockIdx.x;   // 1 & 2
    int maxi = (blockDim.x * blockIdx.x + 1) - 1; // 3 & 4 & 5
    //Allocate local memory
    __shared__ double y_local[LOCALDIM];  // 6
    
    //Copy to local index
    int local_idx = threadIdx.x+1;  // 7 & 8
    y_local[local_idx] = d_y[i]; 
    
    //Check for edge case
    if(i == 0 || i < nx)  {
        d_dvdt[i] = 0;
        d_y[i] = 0;
    } else {
        if(local_idx == 1) {  // 9 & 10  
                y_local[0] = d_y[mini];  
                y_local[LOCALDIM-1] = d_y[maxi]; 
        }
        for(int it=0;it<nt;it++) {
            //fill in edges
            d_dvdt[i]=(y_local[local_idx+1]+y_local[local_idx-1]-2.0*y_local[local_idx])*(dx2inv); 

            d_v[i] = d_v[i] + dt*d_dvdt[i];
            y_local[i] = y_local[i] + dt*y_local[i];
        }
    }
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
    dx = (max-min)/(double)(nx);
    x[0] = min;
    for(i=1;i<nx;i++) {
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

   fprintf(stderr, "dt = %f, dx = %f\n", dt,dx);
   CUDA_CALL(cudaMemcpy(d_y,y,nx*sizeof(double),cudaMemcpyHostToDevice));
   CUDA_CALL(cudaMemcpy(d_v,v,nx*sizeof(double),cudaMemcpyHostToDevice));
   CUDA_CALL(cudaMemcpy(d_dvdt,dvdt,nx*sizeof(double),cudaMemcpyHostToDevice));
    
   dx2inv=1.0/(dx*dx);
   int block_size=BLOCKDIM;
   int block_no = (nx-2)/block_size;
   dim3 dimBlock(block_size,1,1);
   dim3 dimGrid(block_no,1,1);

    for(it=0;it<nt/2;it++) {
        accel_update<<<dimGrid, dimBlock>>>(d_dvdt, d_y, nx, dx2inv);
        //tile_accel_update<<<dimGrid, dimBlock>>>(d_dvdt, d_y, nx, dx2inv);
        pos_update<<<dimGrid, dimBlock>>>(d_dvdt, d_y, d_v, dt);
    }
    
    
    // Include loop in kernel
    //all<<<dimGrid, dimBlock>>>(d_dvdt, d_y, d_v, nx, dx2inv, dt, nt);
  
   CUDA_CALL(cudaMemcpy(y,d_y,nx*sizeof(double),cudaMemcpyDeviceToHost));

    //printf("x, y\n");
    for(i=0; i<nx; i++) {
        printf("%g, %g\n",x[i],y[i]);
    }

    return 0;
}

