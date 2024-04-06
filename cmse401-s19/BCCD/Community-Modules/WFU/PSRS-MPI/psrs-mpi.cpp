// Parallel sorting using regular sampling
// C++ implementation using MPI
//
// David John, August 2013
// (c) Wake Forest University, 2013

#include <iostream>
#include <iomanip>
#include <string>
#include <cstdlib>
#include <algorithm>
#include <string.h>

using namespace std;

#include "mpi.h"
#include "multimerge.h"
#include "utilities.h"

// The main program that is executed on each processor

int main( int argc, char* argv[])
{
   	// processor rank, and total number of processors
    int myid, numprocs;
    
    // for timing used by root processor (#0)
    double startwtime = 0.0, endwtime;


	// *******************************************
	//
	// PHASE I:  Initialization

    // start MPI and pass command line arguments to MPI
    MPI::Init(argc, argv);

    // Collect information about the number of
    // processors and rank of this processor
    numprocs = MPI::COMM_WORLD.Get_size();
    myid = MPI::COMM_WORLD.Get_rank();

	// look through arguments for:
	//		 -DS nnnn to set myDataSize
	//		 -SR nnnn to set randomSeed
	int randomSeed = 1000;
	int myDataSize = 4000;
	for(int i=0;i<argc;i++)
	{
		// search for special arguments
		if (strcmp(argv[i],"-DS")==0)
		{
			myDataSize = atoi(argv[i+1]); i++;
		}
		else if (strcmp(argv[i],"-SR")==0)
		{
			randomSeed = atoi(argv[i+1]); i++;
		}
	}

    // array to hold input data to sort
	// myDataLengths[] and myDataStarts[] used with Scatterv() to distribute
	// myData[] across processors
	int myData[myDataSize];
	int myDataLengths[numprocs];
	int myDataStarts[numprocs];
	
	// communication buffer used for determination of pivot values
	int pivotbuffer[numprocs*numprocs];
	int pivotbufferSize;

	// Compute the individual lengths of mydata[] to be distributed
	// to the numproc processors.  The last processor gets the "extras".
	for(int i=0;i<numprocs;i++)
	{
		myDataLengths[i] = myDataSize/numprocs;
		myDataStarts[i]= i*myDataSize/numprocs;
	}
	myDataLengths[numprocs-1]+=(myDataSize%numprocs);

    // Root node initializes the testing data, and also starts
    // the timer
    if (myid == 0)
    {
		// set random seed and randomly generate testing data
		srandom(randomSeed);
		for(int index=0; index<myDataSize; index++)
		{
			myData[index] = random()% 900;
		}
		
		startwtime = MPI::Wtime();
    }


	// *******************************************
	//
	// PHASE II:  Scatter data, local sorts and collect regular sample

	//  The data is scattered to all processors from the root processor (#0)
	//  (root processor) does "in place".
	if (myid==0)
	{
		MPI::COMM_WORLD.Scatterv(myData,myDataLengths,myDataStarts,MPI::INT,
				MPI_IN_PLACE,myDataLengths[myid],MPI::INT,0);
	}
	else
	{			
		MPI::COMM_WORLD.Scatterv(myData,myDataLengths,myDataStarts,MPI::INT,
				myData,myDataLengths[myid],MPI::INT,0);
	}
      
	// All processors sort their piece of the data using cstdlib::quicksort
	qsort(myData,myDataLengths[myid], sizeof(int), compare_ints);

	// All processors collect regular samples from sorted list
	// Consider an offset to the myData[] index
	for(int index=0;index<numprocs;index++)
	{
		pivotbuffer[index]= myData[index*myDataLengths[myid]/numprocs];
	}


	// *******************************************
	//
	// PHASE III:  Gather and merge samples, and broadcast p-1 pivots

	// The root processor gathers all pivot candidates from the processors, root
	// processor has data "in place"
	if (myid==0)
	{
		MPI::COMM_WORLD.Gather(MPI_IN_PLACE,numprocs,MPI::INT,
			pivotbuffer,numprocs,MPI::INT,0);
	}
	else
	{
		MPI::COMM_WORLD.Gather(pivotbuffer,numprocs,MPI::INT,
			pivotbuffer,numprocs,MPI::INT,0);
	}

	//  Root processor multimerges the lists together and then selects
	//  final pivot values to broadcast
	if (myid == 0)
	{
		// multimerge the numproc sorted lists into one
		int *starts[numprocs];  // array of lists
		int lengths[numprocs];  // array of lengths of lists
		for(int i=0;i<numprocs;i++)
		{
			starts[i]=&pivotbuffer[i*numprocs];
			lengths[i]=numprocs;
		}
		int tempbuffer[numprocs*numprocs];  // merged list
		multimerge(starts,lengths,numprocs,tempbuffer,numprocs*numprocs);

		// regularly select numprocs-1 of pivot candidates to broadcast
		// as partition pivot values for myData
		for(int i=0; i<numprocs-1; i++)
		{
			pivotbuffer[i] = tempbuffer[(i+1)*numprocs];
		}				
	}

	// Root processor (#0) broadcasts the partition values
	MPI::COMM_WORLD.Bcast(pivotbuffer,numprocs-1,MPI::INT,0);


	// *******************************************
	//
	// PHASE IV: Local data partitioned

	// All processors partition their data members based on the
	// pivot values stored in pivotbuffer[].

	// Partition information for myData[]: 
	// 		index of beginning of ith class is classStart[i],
	//		length of ith class is classLength[i], and
	// 		members of ith class, myData[j], have the property
	//   		pivotbuffer[i-1]<= myData[j] < pivotbuffer[i]
	int classStart[numprocs];
	int classLength[numprocs];
	
	// need for each processor to partition its list using the values
	// of pivotbuffer
	int dataindex=0;
	for(int classindex=0; classindex<numprocs-1; classindex++)
	{
		classStart[classindex] = dataindex;
		classLength[classindex]=0;

		// as long as dataindex refers to data in the current class
		while((dataindex< myDataLengths[myid]) 
			&& (myData[dataindex]<=pivotbuffer[classindex]))
		{
			classLength[classindex]++;
			dataindex++;
		}		
	}
	// set Start and Length for last class
	classStart[numprocs-1] = dataindex;
	classLength[numprocs-1] = myDataLengths[myid] - dataindex;
	
	
	// *******************************************
	//
	// PHASE V:  All ith classes are gathered by processor i 
	int recvbuffer[myDataSize];    // buffer to hold all members of class i
	int recvLengths[numprocs];     // on myid, lengths of each myid^th class
	int recvStarts[numprocs];      // indices of where to start the store from 0, 1, ...

	// processor iprocessor functions as the root and gathers from the
	// other processors all of its sorted values in the iprocessor^th class.  
	for(int iprocessor=0; iprocessor<numprocs; iprocessor++)
	{	
		// Each processor, iprocessor gathers up the numproc lengths of the sorted
		// values in the iprocessor class
		MPI::COMM_WORLD.Gather(&classLength[iprocessor], 1, MPI::INT, 
			recvLengths,1,MPI::INT,iprocessor);
	

		// From these lengths the myid^th class starts are computed on
		// processor myid
		if (myid == iprocessor)
		{
			recvStarts[0]=0;
			for(int i=1;i<numprocs; i++)
			{
				recvStarts[i] = recvStarts[i-1]+recvLengths[i-1];
			}
		}

		// each iprocessor gathers up all the members of the iprocessor^th 
		// classes from the other nodes
		MPI::COMM_WORLD.Gatherv(&myData[classStart[iprocessor]],
			classLength[iprocessor],MPI::INT,
			recvbuffer,recvLengths,recvStarts,MPI::INT,iprocessor);
	}
		
	
	// multimerge these numproc lists on each processor
	int *mmStarts[numprocs]; // array of list starts
	for(int i=0;i<numprocs;i++)
	{
		mmStarts[i]=recvbuffer+recvStarts[i];
	}
	multimerge(mmStarts,recvLengths,numprocs,myData,myDataSize);
	
	int mysendLength = recvStarts[numprocs-1] + recvLengths[numprocs-1];
	
	// *******************************************
	//
	// PHASE VI:  Root processor collects all the data


	int sendLengths[numprocs]; // lengths of consolidated classes
	int sendStarts[numprocs];  // starting points of classes
	// Root processor gathers up the lengths of all the data to be gathered
	MPI::COMM_WORLD.Gather(&mysendLength,1,MPI::INT,
		sendLengths,1,MPI::INT,0);

	// The root processor compute starts from lengths of classes to gather
	if (myid == 0)
	{
		sendStarts[0]=0;
		for(int i=1; i<numprocs; i++)
		{
			sendStarts[i] = sendStarts[i-1]+sendLengths[i-1];
		}	
	}

	// Now we let processor #0 gather the pieces and glue them together in
	// the right order
	int sortedData[myDataSize];
	MPI::COMM_WORLD.Gatherv(myData,mysendLength,MPI::INT,
		sortedData,sendLengths,sendStarts,MPI::INT,0);

	// the root processor prints the elapsed clock time
    if (myid == 0)
	{
		endwtime = MPI::Wtime();
        cout << "wall clock time (seconds) = " 
		     << scientific << setprecision(4) << endwtime-startwtime << endl;

		cout << "Data set " << issorted(sortedData,myDataSize) << " sorted:" 
			<< endl;	     
	}
        
    // shutdown MPI on the processor
    MPI::Finalize();
    return 0;
}
