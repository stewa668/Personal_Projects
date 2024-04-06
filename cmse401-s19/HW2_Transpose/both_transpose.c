#include <stdio.h>
#include <stdlib.h>
#define min(X,Y) ((X) < (Y) ? (X) : (Y))
typedef int bool;
#define true 1
#define false 0

int main()
{
   printf("Hello, World!\n");
   int matSize = 10;
   int bsize = 16;
   bool Basic=true;
   //Generate Random Matrix
   srand(0);
   int matA[matSize*matSize];
   int matB[matSize*matSize];
   for (int i = 0; i < matSize*matSize; i++) {
   	matA[i] = rand() % 100;
   }

   //Print out Matrix
   for(int i=0;i<matSize;i++){
        for(int j=0;j<matSize;j++){
		printf("%d\t", matA[i*matSize+j]);
	}
	printf("\n");
   }

   if(Basic==true){
	   //Basic Transpose
   	for(int i=0;i<matSize;i++){
       	 	for(int j=0;j<matSize;j++){
       	 		matB[j*matSize+i]=matA[i*matSize+j];
        	}
   	}
   } else {
   	for(int i=0;i<matSize;i+=bsize){
		for(int j=0;j<matSize;j+=bsize){
			for(int i1=i;i1 < min(i+bsize,matSize); i1++){
				for(int j1=j;j1 < min(j+bsize,matSize); j1++){
					matB[j1*matSize+i1]=matA[i1*matSize+j1];
				}
			}
		}
   	}
    }	

   printf("done Transpose\n");

   for(int i=0;i<matSize;i++){
        for(int j=0;j<matSize;j++){
                printf("%d\t", matB[i*matSize+j]);
        }
        printf("\n");
   }


   return 0;
}

