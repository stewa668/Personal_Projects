#include "omp.h"
#include <stdio.h>
//#include<iostream>
//using std::cout;
int main()
{
    #pragma omp parallel
    {
        int ID = omp_get_thread_num();
	//cout << "hello(" << ID << ") world(" << ID << ") \n" << std::endl;
        printf("hello(%d)",ID);
        printf(" world(%d) \n",ID);
    }
}
