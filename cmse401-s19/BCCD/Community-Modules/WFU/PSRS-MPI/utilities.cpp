
// Utility functions
// David John, August 2013
// (c) Wake Forest University, 2013

#include "utilities.h"


// compare_ints() used by cstdlib::qsort() to compare two
// integers
int compare_ints(const void *a, const void *b)
{
	int myint1 = *reinterpret_cast<const int *>(a);
	int myint2 = *reinterpret_cast<const int *>(b);
	if (myint1<myint2) return -1;
	if (myint1>myint2) return 1;
	return 0;
}

// Utility function to help verify that list is sorted
string issorted(int xxx[], int xxxStart, int xxxLength)
{
	for(int i=xxxStart; i<xxxStart+xxxLength-1;i++)
	{
		if (xxx[i]>xxx[i+1])
			return "is not";
	}

	return "is";
}

string issorted(int xxx[], int xxxLength)
{
	return issorted(xxx,0,xxxLength);
}

// ----------------

// Utility to show array values
void dumpArray(int myid, string arrayName, int array[], int start, int length)
{
	for(int i=start;i<start+length;i++)
	{
		cout << myid << ": " << arrayName << "[" << i << "] = " << array[i] << endl;
	}
	return;
}

void dumpArray(int myid, string arrayName, int array[], int length)
{ 
	dumpArray(myid, arrayName, array, 0, length);
	return;
}
 
