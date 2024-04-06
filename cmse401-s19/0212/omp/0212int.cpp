#include<iostream>
using std::cin; using std::cout; using std::endl;
#include<omp.h>


static long num_steps = 1000000; // race condition...
double step;
int nthreads = 16 ;
int main()
{
    //cin >> num_steps ;
    double pi=0.0;
    step = 1.0/(double) num_steps;
    omp_set_num_threads(nthreads);
    #pragma omp parallel
    { 
    double sum, x;
         int ID = omp_get_thread_num() ;
        for (int i=ID ; i<num_steps; i += nthreads) 
        {
            x = (i + 0.5) * step;
            sum += 4.0/(1.0+x*x);
        }
        #pragma omp critical
        pi += sum*step ;
    }
    cout.precision(17) ;
    cout << pi << " " << (pi-3.14159265358979324) << endl ;
}
