#include<iostream>
using std::cin; using std::cout; using std::endl;

static long num_steps ;// = 100000;
double step;
int main()
{
    cin >> num_steps ;
    int i; double x,pi,sum=0.0;
    step = 1.0/(double) num_steps;
    
    for (i=0;i<num_steps;i++) 
    {
        x = (i + 0.5) * step;
        sum = sum+4.0/(1.0+x*x);
    }
    pi = step * sum;
    cout.precision(17) ;
    cout << pi << " " << (pi-3.14159265358979324) << endl ;
}
