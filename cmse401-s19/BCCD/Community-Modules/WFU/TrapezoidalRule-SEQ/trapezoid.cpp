#include <cmath>
#include <ctime>
#include <iostream>
using namespace std;

// Aunction to implement integration of f(x) over the interval
// [a,b] using the trapezoid rule with nsub subdivisions.
float trapezoid(float a, float b, int nsub, float (*f)(float) )
{
	// initialize the partial sum to be f(a)+f(b) and
	// deltaX to be the step size using nsub subdivisions
	float psum = f(a)+f(b);
	float deltaX = (b-a)/nsub;

	// increment the partial sum
	for(int index=1;index<nsub;index++)
	{
		psum = psum + 2.0*(*f)(a+index*deltaX);
	}

	// multiply the sum by the constant deltaX/2.0
	psum = (deltaX/2.0)*psum;


	// return approximation
	return psum;

}


// Function definition, must be defined over [a,b]
float myfunction(float x)
{

	return exp(-x*x+sin(x));
}

// utility function to convert the length of time into
// milliseconds
double diffclock(clock_t clock1, clock_t clock2)
{
	double diffticks = clock1-clock2;
	double diffms = diffticks/(CLOCKS_PER_SEC/1000);
	return diffms;
}


// Simple main program to call and test
int main()
{



	// Get number of subdivisions
	float Nsubs;
	cout << "Enter the number of subdivisions: ";
	cin >> Nsubs;


	clock_t start = clock();

	// Integral of x*x-2*x+1 over [1.0,2.0]
	float answer= trapezoid(1.0,2.0,Nsubs,&myfunction);


	clock_t stop = clock();


	cout << "The answer is " << answer << endl;
	cout << "Computation time: " << diffclock(stop,start);
	cout << " milliseconds" << endl;

	return 0;
}


