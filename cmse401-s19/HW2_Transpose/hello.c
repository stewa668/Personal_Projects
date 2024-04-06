#include <stdio.h>
int main()
{
   printf("Hello, World!\n");
   return 0;
}

/*do i=1,matSize, bsize 
    do j=1,matSize, bsize
        do i1=i,min(i+bsize-1,matSize) 
            do j1=j,min(j+bsize-1,matSize)
                matB(i1,j1) = matA(j1,i1) 
            enddo
        enddo 
    enddo
enoddo
*/
