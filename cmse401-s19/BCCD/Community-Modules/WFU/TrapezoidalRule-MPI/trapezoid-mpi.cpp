
#include <iostream>
#include <iomanip>
#include <string>
#include <cmath>

using namespace std;

#include "mpi.h"

// Function to implement integration of f(x) over the interval
// [a,b] using the trapezoid rule with nsubs subdivisions.
double trapezoid(double a, double b, int nsubs, double (*f)(double) )
{
	// initialize the partial sum to be f(a)+f(b) and
	// deltaX to be the step size using n subdivisions
	double psum = f(a)+f(b);
	double deltaX = (b-a)/nsubs;

	// increment the partial sum
	for(int index=1;index<nsubs;index++)
	{	
			psum = psum + 2.0*f(a+index*deltaX);
	}

	// multiply the sum by the constant deltaX/2.0
	psum = (deltaX/2.0)*psum;


	// return approximation
	return psum;

}



// The function to be integrated

double myfunction( double x )
{
    return  exp(-x*x+sin(x));
}



//
// The main program that is executed on each
// processor

int main( int argc, char* argv[])
{
	double a,b;
	int n;
    int myid, numprocs;
    double psum=0.0, sum=0.0;

    // for timing
    double startwtime, endwtime;

    // pass command line arguments to MPI
    MPI::Init(argc, argv);

    // Collect information about the number of
    // processors and names of the processors
    numprocs =MPI::COMM_WORLD.Get_size();
    myid = MPI::COMM_WORLD.Get_rank();

	a=1.0;
	b=2.0;

    // Root node sets the number of subdivisions used
    // in the evaluation of the integral, also start
    // the timer
    if (myid == 0)
      {
		cout << "Enter the number of subdivisions: ";
        cin >> n;
	    startwtime = MPI::Wtime();
       }

     // Broadcast from the root nodes to all others the
     // number of subdivisions
     MPI::COMM_WORLD.Bcast(&n, 1, MPI::INT, 0);
		

      // each processor (including 0) computes its share of sums
	  double xLeft = a+myid*(b-a)/numprocs;
	  double xRight = a+(myid+1)*(b-a)/numprocs;
      psum = trapezoid(xLeft,xRight,
                       n/numprocs,&myfunction);
      


	  // the processors add together their partial sum values
	  // into the sum and give the answer to processor 0 (rendezvous)
      MPI::COMM_WORLD.Reduce(&psum, &sum, 1, MPI::DOUBLE, MPI::SUM, 0);

	  // the root processor prints the sum, the "error", and
	  // the elapsed clock time
      if (myid == 0)
	    {
		   endwtime = MPI::Wtime();
	       cout << sum << endl;
           cout << "wall clock time (milliseconds) = " 
		      << (int)((endwtime-startwtime)*1000) << endl;	       
	    }
        
    

    // shutdown MPI on the processor
    MPI::Finalize();

    return 0;
}
