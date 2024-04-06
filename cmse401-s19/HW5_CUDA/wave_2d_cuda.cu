#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "png_util.h"
#define min(X,Y) ((X) < (Y) ? (X) : (Y))
#define max(X,Y) ((X) > (Y) ? (X) : (Y))
#define CUDA_CALL(x) {cudaError_t cuda_error__ = (x); if (cuda_error__) { fprintf(stderr, "CUDA error: " #x " returned \"%s\"\n", cudaGetErrorString(cuda_error__)); fflush(stderr); exit(cuda_error__); } }


__global__ void accel_update(double* a_d, double* z_d, int nx, int ny, double dx2inv, double dy2inv)
{
    long i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i <  nx || i >= (ny-1)*nx) // Up Down
        a_d[i] = 0;
    else if (i%nx == 0 || i%nx == nx-1) //Left Right
        a_d[i] = 0 ;
    else {
        double ax = (z_d[i-1]+z_d[i+1]-2.0*z_d[i])*dx2inv;
        double ay = (z_d[i-nx]+z_d[i+nx]-2.0*z_d[i])*dy2inv;
        a_d[i] = (ax+ay)/2;            
    }       
}

__global__ void z_update(double* a_d, double* v_d, double* z_d, double dt)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    v_d[i] = v_d[i] + dt*a_d[i];
    __syncthreads();
    z_d[i] = z_d[i] + dt*v_d[i];
}

int main(int argc, char ** argv) {
//double total = 0, setup = 0, sim = 0, io = 0 ;
//struct timeval tot0, tot1, sim0, sim1, set1, io0, io1;
//gettimeofday(&tot0,0);

    int nx = 500;
    int ny = 500;
    int nt = 20000; 
    int frame=0;
    //int nt = 1000000;
    int r,c,it;
    double dx,dy,dt;
    double max,min;
    double tmax;
    double dx2inv, dy2inv;
    char filename[sizeof "./images/cuda00000.png"];

    image_size_t sz; 
    sz.width=nx;
    sz.height=ny;

    //make mesh
    double * h_z = (double *) malloc(nx*ny*sizeof(double));
    double ** z = (double **) malloc(ny * sizeof(double*));
    for (r=0; r<ny; r++){
    	z[r] = &h_z[r*nx];
    }

    //Velocity
    double * h_v = (double *) malloc(nx*ny*sizeof(double));
    double ** v = (double **) malloc(ny * sizeof(double*));
    for (r=0; r<ny; r++)
        v[r] = &h_v[r*nx];

    //Accelleration
    double * h_a = (double *) malloc(nx*ny*sizeof(double));
    double ** a = (double **) malloc(ny * sizeof(double*));
    for (r=0; r<ny; r++)
        a[r] = &h_a[r*nx];

    //output image
    unsigned char * o_img = (unsigned char *) malloc(sz.width*sz.height*sizeof(unsigned char));
    unsigned char **output = (unsigned char **) malloc(sz.height * sizeof(unsigned char*));
    for (int r=0; r<sz.height; r++)
        output[r] = &o_img[r*sz.width];

    double * z_d;
    CUDA_CALL(cudaMalloc((void **)&z_d,(nx*ny*sizeof(double))));
    double * v_d;
    CUDA_CALL(cudaMalloc((void **)&v_d,(nx*ny*sizeof(double))));
    double * a_d;
    CUDA_CALL(cudaMalloc((void **)&a_d,(nx*ny*sizeof(double))));
    //double * o_d;
    //CUDA_CALL(cudaMalloc((void **)&o_d,(sz.width*sz.height*sizeof(unsigned char))));

    max=10.0;
    min=0.0;
    dx = (max-min)/(double)(nx-1);
    dy = (max-min)/(double)(ny-1);
    
    tmax=40.0;
    dt= (tmax-0.0)/(double)(nt-1);

    double x,y; 
    for (r=0;r<ny;r++)  {
    	for (c=0;c<nx;c++)  {
		x = min+(double)c*dx;
		y = min+(double)r*dy;
        	z[r][c] = exp(-(sqrt((x-5.0)*(x-5.0)+(y-5.0)*(y-5.0))));
        	v[r][c] = 0.0;
	        a[r][c] = 0.0;
    	}
    }
    
    dx2inv=1.0/(dx*dx);
    dy2inv=1.0/(dy*dy);

    //Set up blocks
    //block_size=32;
    dim3 dimBlock(ny,1,1);
    dim3 dimGrid(nx,1,1);

    CUDA_CALL(cudaMemcpy(z_d, h_z, nx*ny*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(v_d, h_v, nx*ny*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(a_d, h_a, nx*ny*sizeof(double), cudaMemcpyHostToDevice));

//gettimeofday(&set1,0) ;
//setup = (set1.tv_sec-tot0.tv_sec)*1000000 + set1.tv_usec-tot0.tv_usec;


    for(it=0;it<nt-1;it++) {
//gettimeofday(&sim0,0) ;

	//aup;accel_update<<<dimGrid,dimBlock>>>(a_d, z_d, nx, ny, dx2inv, dy2inv);
        accel_update<<<dimGrid,dimBlock>>>(a_d, z_d, nx, ny, dx2inv, dy2inv);



//        for (r=1;r<ny-1;r++)  
//    	    for (c=1;c<nx-1;c++)  {
//		double ax = (z[r+1][c]+z[r-1][c]-2.0*z[r][c])*dx2inv;
//		double ay = (z[r][c+1]+z[r][c-1]-2.0*z[r][c])*dy2inv;
//		a[r][c] = (ax+ay)/2;
//	    }
        //vzup
	z_update<<<dimGrid,dimBlock>>>(a_d, v_d, z_d, dt);
//        for (r=1;r<ny-1;r++)  
//    	    for (c=1;c<nx-1;c++)  {
//               v[r][c] = v[r][c] + dt*a[r][c];
//               z[r][c] = z[r][c] + dt*v[r][c];
//            }

//gettimeofday(&sim1,0);
//sim += (sim1.tv_sec-sim0.tv_sec)*1000000 + sim1.tv_usec-sim0.tv_usec;


	if (it % 100 ==0)
	{

//gettimeofday(&sim0, 0) ;
            CUDA_CALL(cudaMemcpy(h_z, z_d, nx*ny*sizeof(double), cudaMemcpyDeviceToHost));
    	    double mx,mn;
    	    mx = -999999;
            mn = 999999;
            for(r=0;r<ny;r++)
                for(c=0;c<nx;c++){
           	    mx = max(mx, z[r][c]);
           	    mn = min(mn, z[r][c]);
        	}
    	    for(r=0;r<ny;r++)
                for(c=0;c<nx;c++)
                    output[r][c] = (unsigned char) round((z[r][c]-mn)/(mx-mn)*255);
//gettimeofday(&sim1,0);
//sim += (sim1.tv_sec-sim0.tv_sec)*1000000 + sim1.tv_usec-sim0.tv_usec;
//gettimeofday(&io0,0) ;
    	    sprintf(filename, "./images/cuda%05d.png", frame);
            printf("Writing %s\n",filename);    
    	    write_png_file(filename,o_img,sz);
	    frame+=1;
//gettimeofday(&io1,0);
//io += (io1.tv_sec-io0.tv_sec)*1000000 + io1.tv_usec-io0.tv_usec; ;
        }

    }
//gettimeofday(&sim0,0);
    CUDA_CALL(cudaMemcpy(h_z, z_d, nx*ny*sizeof(double), cudaMemcpyDeviceToHost));
    
    double mx,mn;
    mx = -999999;
    mn = 999999;
    for(r=0;r<ny;r++)
        for(c=0;c<nx;c++){
	   mx = max(mx, z[r][c]);
	   mn = min(mn, z[r][c]);
        }
//gettimeofday(&sim1,0);
//sim += (sim1.tv_sec-sim0.tv_sec)*1000000 + sim1.tv_usec-sim0.tv_usec;
//gettimeofday(&io0,0) ;

    printf("%f, %f\n", mn,mx);

//gettimeofday(&io1,0);
//io += (io1.tv_sec-io0.tv_sec)*1000000 + io1.tv_usec-io0.tv_usec;
//gettimeofday(&sim0,0);

    for(r=0;r<ny;r++)
        for(c=0;c<nx;c++){  
	   output[r][c] = (char) round((z[r][c]-mn)/(mx-mn)*255);  
	}
//gettimeofday(&sim1,0);
//sim += (sim1.tv_sec-sim0.tv_sec)*1000000 + sim1.tv_usec-sim0.tv_usec;
//gettimeofday(&io0,0);

    sprintf(filename, "./images/cuda%05d.png", it);
    printf("Writing %s\n",filename);    
    //Write out output image using 1D serial pointer
    write_png_file(filename,o_img,sz);
//gettimeofday(&io1,0);
//io += (io1.tv_sec-io0.tv_sec)*1000000 + io1.tv_usec-io0.tv_usec;
//gettimeofday(&tot1,0);
//total = (tot1.tv_sec-tot0.tv_sec)*1000000 + tot1.tv_usec-tot0.tv_usec;
//printf("Total time %f\n" ,total/1000000);
//printf("Setup time %f\n" , setup/1000000);
//printf("Simulation time %f\n" ,sim/1000000);
//printf("I/O time %f\n" ,io/1000000);

    return 0;
}
