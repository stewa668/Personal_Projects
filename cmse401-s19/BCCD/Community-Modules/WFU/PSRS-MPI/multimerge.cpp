// multimerge.cpp
// David John
// (c) Wake Forest University, 2013

// Function to merge Number sorted arrays into one sorted array
//
// The addresses and lengths of the sorted arrays are passed via *start[] and lengths[]
//    start[j][k]  represents the kth value in the ith array


#include "multimerge.h"

// This structure is what is placed in the priority queue: an index to the 
//		appropriate array,  an index to an element in the appropriate array, and 
//		the value of stored at the index of the element in the appropriate array
struct mmdata 
{
	int stindex;
	int index;
	int stvalue;

	mmdata(int st=0, int id=0, int stv = 0):stindex(st),index(id),stvalue(stv){}

};


// comparison operator
bool operator<( const mmdata & One, const mmdata & Two)
{
	return One.stvalue > Two.stvalue;
}

int multimerge(int * starts[], const int lengths[], const int Number, 
			   int newArray[], const int newArrayLength)
{
 	// Create priority queue.  There will be at most one item in the priority queue
 	// for each of the Number lists.
 	priority_queue< mmdata> priorities;

 	// Examine each of the Number start[] lists, place the first location into 
	// the priority 	queue if the list is not empty
 	for(int i=0; i<Number;i++)
 	{
		if (lengths[i]>0)
		{
			priorities.push(mmdata(i,0,starts[i][0]));
		}
	}


	// As long as priorities is not empty, pull off the top member (the smallest 
	//value from list i), push it into the newArray, and place the next element from 
	// list i in the priority queue
	int newArrayindex = 0;  // index into the merged array
	while (!priorities.empty() && (newArrayindex<newArrayLength))
	{
		// grab the smallest element, and remove it from the priority queue
		mmdata xxx = priorities.top();
		priorities.pop();

		// insert this smallest element into the merged array
		newArray[newArrayindex++] = starts[xxx.stindex][xxx.index];

		// if start[xxx.stindex] is not empty, place the next member into priority
		if ( lengths[xxx.stindex]>(xxx.index+1))
		{
			priorities.push(mmdata(xxx.stindex, xxx.index+1, 
								starts[xxx.stindex][xxx.index+1]));
		}
}

// return the logical size of the merged array
return newArrayindex;
}

