#include<iostream>
using std::cin; using std::cout; using std::endl;
#include<omp.h>


static long num_steps = 1000000; // race condition...
double step;
int main()
{
    //cin >> num_steps ;
    int i; double pi=0.0;
    step = 1.0/(double) num_steps;
    omp_set_num_threads(16);
    double sum [16] ;
    #pragma omp parallel
    {
         double x;
         int ID = omp_get_thread_num() ;
        for (int i=ID ; i<num_steps; i += 16)
        {
            x = (i + 0.5) * step;
            sum[ID] += 4.0/(1.0+x*x);
        }
    }
    for (int i=0 ; i<16; i++){
       pi += sum[i] * step ;
    }
    cout.precision(17) ;
    cout << pi << " " << (pi-3.14159265358979324) << endl ;
}

