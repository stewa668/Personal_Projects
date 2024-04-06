#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main(int argc, char ** argv) {
    int nxx = 5000;
    double x[nxx];
    double y[nxx];
    double v[nxx];
    double dvdt[nxx];
    #pragma omp parallel default(none) shared(x, y, v, dvdt) 
    {
    int nx = 5000;
    int nt = 1000000;
    int i,it;
    double dt;
    double dx;
    double max,min;
    double dx2inv;
    double tmax;
    int nxm1;

    max=10.0;
    min=0.0;
    dx = (max-min)/(double)(nx-1);
    x[0] = min;
    #pragma omp for schedule(static) 	   
    for(i=1;i<nx-1;i++) {
        x[i] = min+(double)i*dx;
    }
    x[nx-1] = max;
    tmax=10.0;
    dt= (tmax-0.0)/(double)(nt-1);
    
    #pragma omp for schedule(static) 	   
    for (i=0;i<nx;i++)  {
        y[i] = exp(-(x[i]-5.0)*(x[i]-5.0));
        v[i] = 0.0;
        dvdt[i] = 0.0;
    }
    
    dx2inv=1.0/(dx*dx);
    nxm1=nx-1;
	for(it=0;it<nt-1;it++) {
    	#pragma omp for schedule(static)	   
        for(i=1;i<nxm1;i+=1)
	{
            dvdt[i]=(y[i+1]+y[i-1]-2.0*y[i])*(dx2inv);
            //dvdt[i+1]=(y[i+2]+y[i]-2.0*y[i+1])*(dx2inv);
	}
    	#pragma omp for schedule(static)	   
        for(i=1; i<nxm1; i++)  {
            v[i] = v[i] + dt*dvdt[i];
            y[i] = y[i] + dt*v[i];
        }
        }
   } 
    for(int i=nxx/2-10; i<nxx/2+10; i++) {
        printf("%g %g\n",x[i],y[i]);
    }

    return 0;
}
