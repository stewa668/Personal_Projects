
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

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
    
    for(it=0;it<200000;it++) {
        for(i=1;i<nx-1;i++)
            dvdt[i]=(y[i+1]+y[i-1]-2.0*y[i])*(dx2inv);

        for(i=1; i<nx-1; i++)  {
            v[i] = v[i] + dt*dvdt[i];
            y[i] = y[i] + dt*v[i];
        }

    }

    for(i=0; i<nx; i++) {
        printf("%g, %g\n",x[i],y[i]);
    }
    
    return 0;
}
