/* Needed for printing */ 
#include <stdio.h>          
#include <stdlib.h>

/* Get the MPI header file */
#include <mpi.h>
#include <unistd.h>

/* Max number of nodes to test */
#define max_nodes 264  

/* Largest hostname string hostnames */
#define str_length 50       
int main(int argc, char **argv)
{
   /* Declare variables */
   int   proc, rank, size, namelen;
   int   ids[max_nodes];
   char  hostname[str_length][max_nodes];
   char  p_name[str_length];

   MPI_Status status;
   MPI_Init(&argc, &argv);
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   MPI_Comm_size(MPI_COMM_WORLD, &size);
   MPI_Get_processor_name(p_name,&namelen);
if (rank==0) {
   printf("Hello From: %s I am the receiving processor %d of %d\n",p_name, rank+1, size);
   for (proc=1;proc<size;proc++) {
      MPI_Recv(&hostname[0][proc], str_length,MPI_INT,proc, 1,MPI_COMM_WORLD,&status);
      MPI_Recv(&ids[proc], str_length,MPI_INT,proc, 2,MPI_COMM_WORLD,&status);
      printf("Hello From: %-20s I am processor %d of %d\n", &hostname[0][proc], ids[proc]+1, size);
   }
} else { // NOT Rank 0
      srand(rank);
      int t = rand()%10+1;
      sleep(t);
      MPI_Send(&p_name,str_length, MPI_INT,0,1,MPI_COMM_WORLD);
      MPI_Send(&rank,str_length, MPI_INT,0,2,MPI_COMM_WORLD);
   }
   MPI_Finalize();

   return(0);
}
