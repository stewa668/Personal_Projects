#include <iostream>
using namespace std;
#include <mpi.h>

#ifdef STAT_KIT
#include "../StatKit/petakit/pkit.h"    // For PetaKit output
#endif


int main(int argc, char *argv[]) {
  MPI::Init(argc, argv);

#ifdef STAT_KIT
         startTimer();
#endif
  
  int rank = MPI::COMM_WORLD.Get_rank();
  int size = MPI::COMM_WORLD.Get_size();
  
  cout << "Hello World! I am " << rank << " of " << size << endl;
  
  MPI::Finalize();

#ifdef STAT_KIT
        printStats("Hello World MPI CPP",size,"mpi",1, "1", 0, 0);
#endif
}
