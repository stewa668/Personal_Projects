#include<iostream>
#include<omp.h>
#include <chrono>
#include<iomanip>
using namespace std::chrono;

static long num_steps  = 100000;
double step;
int main()
{
    auto start = high_resolution_clock::now();
    int i; double pi,sum=0.0;
    step = 1.0/(double) num_steps;
    omp_set_num_threads(16);
    #pragma omp parallel 
    {
    double x;
    #pragma omp for reduction(+:sum)
    for (i=0;i<num_steps;i++) 
    {
        x = (i + 0.5) * step;
        sum = sum+4.0/(1.0+x*x);
    } 
    }
    pi = step * sum;
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    std::cout << std::setprecision(15);
    std::cout << pi << " " << duration.count() << std::endl;
}
