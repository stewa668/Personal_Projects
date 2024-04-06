#include <stdio.h>
#include <stdlib.h>
#define min(X,Y) ((X) < (Y) ? (X) : (Y))

//from https://stackoverflow.com/questions/1921539/using-boolean-values-in-c
typedef int bool;
#define true 1
#define false 0

int main(int argc, char** argv)
{
   printf("usage: transpose matSize bsize Basic Output\n", argc);
   int matSize = 10;
   int bsize = 16;
   bool Basic=true;
   bool Output=false;
   
   if (argc > 1){
	matSize = (int) strtol(argv[1], NULL , 10); 
   }
   if (argc > 2){
	bsize = (int) strtol(argv[2], NULL , 10); 
   }
   if (argc > 3){
	Basic = (bool) strtol(argv[3], NULL , 10); 
   }
   if (argc > 4){
	Output = (bool) strtol(argv[4], NULL , 10);
   }
   printf("usage: transpose %d %d %d %d\n", matSize, bsize, Basic, Output);

   //Generate Random Matrix
   srand(0);
   int matA[matSize*matSize];
   int matB[matSize*matSize];
   for (int i = 0; i < matSize*matSize; i++) {
   	matA[i] = rand() % 100;
   }

   //Print out Matrix
   if (Output) {
      for(int i=0;i<matSize;i++){
           for(int j=0;j<matSize;j++){
   		printf("%d\t", matA[i*matSize+j]);
    	   }
	   printf("\n");
      }
   }

   if(Basic==true){
	printf("Basic Transpose\n");
   	for(int i=0;i<matSize;i++){
       	 	for(int j=0;j<matSize;j++){
       	 		matB[j*matSize+i]=matA[i*matSize+j];
        	}
   	}
   } else {
	printf("Blocked Transpose\n");
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

   printf("Transpose Complete\n");

   if(Output){
   for(int i=0;i<matSize;i++){
        for(int j=0;j<matSize;j++){
                printf("%d\t", matB[i*matSize+j]);
        }
        printf("\n");
   }
   }


   return 0;
}

