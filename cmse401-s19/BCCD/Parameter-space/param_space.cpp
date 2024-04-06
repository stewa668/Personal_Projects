///////////////////////////////////////////
// Dart Parameter Space Study
// Copyright 1997-2002
// David A. Joiner and
//   The Shodor Education Foundation, Inc.
// $Id: param_space.cpp,v 1.5 2012/06/27 19:07:15 ibabic09 Exp $
// This file is part of BCCD, an open-source live CD for computational science
// education.
// 
// Copyright (C) 2010 Andrew Fitz Gibbon, Paul Gray, Kevin Hunter, Dave Joiner, 
//   Sam Leeman-Munk, Tom Murphy, Charlie Peck, Skylar Thompson, & Aaron Weeden 

// 
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
// 
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
// 
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.
///////////////////////////////////////////

#include <mpi.h>
#include "do_stuff.h"
#include "param_space.h"
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <stdio.h>

int main(int argc, char ** argv) {

#ifdef STAT_KIT
	startTimer();
#endif
    
    MPI_Status status;
    int rank, size;

    /* set up MPI process */
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);


    // defaults
    int num_theta = 550;
    int num_rad=50;
    int num_average=100;
    int do_display=1;
    
    // command line arguments
    if (argc > 1) {
        sscanf(argv[1],"%d",&num_rad);
    }
    if (argc > 2) {
        sscanf(argv[2],"%d",&num_theta);
    }
    if (argc > 3) {
        sscanf(argv[3],"%d",&num_average);
    }
    if (argc > 4) {
        sscanf(argv[4],"%d",&do_display);
    }
    if (do_display!=0) do_display=1;


    init_rand(rank);
    double theta_min = 0.0;
    double theta_max = 4.0*asin(1.0);
    double rad_max=1.0;
    double rad_min=0.0;
    
    double sigma=0.5;
    
    
    double * theta = create_grid(num_theta,theta_min,theta_max);

    // (n-1)/p per processor, with (n-1)%p left over.
    int * grid_start = new int[size];
    int grid_finish;
    int distrib = (num_theta-1)%size;
    grid_start[0]=0;
    for (int i=1;i<size;i++) {
        grid_start[i]=grid_start[i-1]+((num_theta-1)/size);
        if (distrib-- > 0) grid_start[i]++;
    }
    
    if (rank==size-1) {
        grid_finish = num_theta-1;
    } else {
        grid_finish = grid_start[rank+1];
    }
    
    do_stuff (rank,size,num_average, grid_finish-grid_start[rank],
        theta[grid_start[rank]],theta[grid_finish],
        num_rad,rad_min,rad_max,sigma, do_display);
      
    delete theta;
    delete grid_start;
    
    /* exit program */
    MPI_Finalize();


#ifdef STAT_KIT
 	printStats("Parameter Space",size,"mpi",num_average, "1", 0, 0);
#endif

    return 0;
}

/* seed random number generator */
void init_rand(int n) {
    time_t my_time;
    time(&my_time);
    srand((long)my_time+(long)n);
}
