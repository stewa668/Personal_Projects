/* DESCRIPTION: Parallel code for approximating the area under a curve using a 
 *                  left Riemann sum.
 * AUTHOR:      Aaron Weeden, Shodor Education Foundation, Inc.
 * DATE:        September 2011
 *              Revised December 2011
 *
 * EXAMPLE USAGE:
 *        To run a program (e.g. serial) with the default domain width and 
 *          default number of rectangles, use:    ./area.serial
 *        To run a program (e.g. openmp) with the default domain width and 
 *          100000 rectangles, use:    ./area.openmp -n 100000
 *        To run a program (e.g. mpi) with a domain from 100.0 to 200.0 and the
 *          default number of rectangles, use:   ./area.mpi -l 100.0 -r 200.0
 *        To run a program (e.g. hybrid) with a domain from 50.1 to 75.5 and
 *          with 50000 rectangles, use:   ./area.hybrid -l 50.1 -r 75.5 -n 50000
 *        To run a program (e.g. serial) with the default left x-boundary of the
 *          domain and 500.0 as the right x-boundary of the domain with the
 *          default number of rectangles, use:  ./area.serial -r 500.0
 */

/***********************
 * Libraries to import *
 ***********************/
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#ifdef _MPI
    #include <mpi.h>
#endif

#ifdef OMP
    #include <omp.h>
#endif

#ifdef STAT_KIT
#include "../StatKit/petakit/pkit.h"    // For PetaKit output
#endif

/***************************************************************************
 * Mathematical function under whose curve we wish to approximate the area *
 ***************************************************************************/
double FUNC(double x)
{
    return x * x;
}

/******************************************
 * MAIN function -- execution starts here *
 ******************************************/
int main(int argc, char** argv)
{

#ifdef STAT_KIT
         startTimer();
#endif


    /* Declare the constants, variables, and data structures */
    int OUR_RANK = 0, OUR_NUMBER_OF_THREADS = 1, OUR_NUMBER_OF_RECTANGLES = 10, 
        NUMBER_OF_RECTANGLES = 10, NUMBER_OF_PROCESSES = 1, MY_THREAD_NUM = 0, 
        my_current_rectangle_id = 0;
    double OUR_X_LEFT = 0.0, X_LEFT = 0.0, X_RIGHT = 10.0, WIDTH = 0.0, 
           RECTANGLE_WIDTH = 0.0, our_total_sum = 0.0, the_total_sum = 0.0, 
           my_current_rectangle_left = 0.0, my_current_rectangle_height = 0.0;
    double * our_areas;
    int our_cardinality = 0; 

    /* For command line parsing */
    int c;

    #ifdef _MPI
        /* Initialize the MPI environment */
        MPI_Init(&argc, &argv);
        MPI_Comm_rank(MPI_COMM_WORLD, &OUR_RANK);
        MPI_Comm_size(MPI_COMM_WORLD, &NUMBER_OF_PROCESSES);
	our_cardinality += NUMBER_OF_PROCESSES; 
    #endif

    /* Parse command line arguments */
    while((c = getopt(argc, argv, "n:l:r:")) != -1)
    {
        switch(c)
        {
            case 'n':
                NUMBER_OF_RECTANGLES = atoi(optarg);
                break;
            case 'l':
                X_LEFT = atof(optarg);
                break;
            case 'r':
                X_RIGHT = atof(optarg);
                break;
            case '?':
            default:
                #ifdef _MPI
                    fprintf(stderr, "Usage: mpirun -np NUMBER_OF_PROCESSES %s [-n NUMBER_OF_RECTANGLES] [-l X_LEFT] [-r X_RIGHT]\n", argv[0]);
                #else
                    fprintf(stderr, "Usage: %s [-n NUMBER_OF_RECTANGLES] [-l X_LEFT] [-r X_RIGHT]\n", argv[0]);
                #endif
                exit(-1);
        }
    }
    argc -= optind;
    argv += optind;

    /* Allocate the array. */
    our_areas = (double*)malloc(NUMBER_OF_RECTANGLES * sizeof(double));

    #ifdef OMP
    #pragma omp parallel
	{
        MY_THREAD_NUM = omp_get_thread_num();
	our_cardinality += omp_get_num_threads();
    #endif

        if(MY_THREAD_NUM == 0)
        {
            /* Calculate the overall width of the domain of the
             *  function and the width of a rectangle.
             */
            WIDTH = X_RIGHT - X_LEFT;
            RECTANGLE_WIDTH = WIDTH / NUMBER_OF_RECTANGLES;
        }
        if(MY_THREAD_NUM == 1 % OUR_NUMBER_OF_THREADS)
        {
            /* Calculate the number of rectangles for 
             *  which the process is responsible. */
            OUR_NUMBER_OF_RECTANGLES = NUMBER_OF_RECTANGLES /
                NUMBER_OF_PROCESSES;
            if(OUR_RANK == NUMBER_OF_PROCESSES - 1)
                OUR_NUMBER_OF_RECTANGLES += NUMBER_OF_RECTANGLES % 
                    NUMBER_OF_PROCESSES;
        }
        if(MY_THREAD_NUM == 2 % OUR_NUMBER_OF_THREADS)
        {
            /* Calculate the left x-boundary of the 
             *  process. */
            OUR_X_LEFT = (OUR_RANK * (NUMBER_OF_RECTANGLES / 
                    NUMBER_OF_PROCESSES)) + X_LEFT;
        }
    #ifdef OMP
    } /* pragma omp parallel */
    #endif

    #ifdef OMP
    #pragma omp parallel for private(my_current_rectangle_id, my_current_rectangle_left, my_current_rectangle_height)
    #endif
    for(my_current_rectangle_id = 0; my_current_rectangle_id < 
            OUR_NUMBER_OF_RECTANGLES; my_current_rectangle_id++)
    {
        /* Calculate the x-value of the left side of the rectangle */
        my_current_rectangle_left = OUR_X_LEFT + my_current_rectangle_id 
            * RECTANGLE_WIDTH;

        /* Calculate the height of the rectangle */
        my_current_rectangle_height = FUNC(my_current_rectangle_left);

        /* Calculate the area of the rectangle */
        our_areas[my_current_rectangle_id] = RECTANGLE_WIDTH * 
                    my_current_rectangle_height;
    }

    /* Calculate the total sum for the process */
    for(my_current_rectangle_id = 0; my_current_rectangle_id < 
            OUR_NUMBER_OF_RECTANGLES; my_current_rectangle_id++)
    {
        our_total_sum += our_areas[my_current_rectangle_id];
    }

    /* Calculate the overall total sum */
    #ifdef _MPI
        MPI_Reduce(&our_total_sum, &the_total_sum, 1, 
                    MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    #else
        the_total_sum = our_total_sum;
    #endif
    
	/* Print the total sum */
	if (OUR_RANK == 0)
		printf("%f\n", the_total_sum);

    /* Deallocate the array */
    free(our_areas);

    #ifdef _MPI
        /* Finalize the MPI environment */
        MPI_Finalize();
    #endif

#ifdef STAT_KIT
	if (our_cardinality == 0)
		our_cardinality = 1; 
        printStats("Are Under Curve",our_cardinality,"mpi",NUMBER_OF_RECTANGLES, "1", 0, 0);
#endif

    /* The code has finished executing successfully. */
    return 0;
}
