/*
  * $Id: device-query-mpi.c,v 1.3 2012/05/01 13:53:22 charliep Exp $
  * 
  * This file is part of BCCD, an open-source live CD for computational science
  * education.
  * 
  * Copyright (C) 2010 Andrew Fitz Gibbon, Paul Gray, Kevin Hunter, Dave 
  *   Joiner, Sam Leeman-Munk, Tom Murphy, Charlie Peck, Skylar Thompson, 
  *   & Aaron Weeden 
  * 
  * This program is free software: you can redistribute it and/or modify
  * it under the terms of the GNU General Public License as published by
  * the Free Software Foundation, either version 3 of the License, or
  * (at your option) any later version.
  * 
  * This program is distributed in the hope that it will be useful,
  * but WITHOUT ANY WARRANTY; without even the implied warranty of
  * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  * GNU General Public License for more details.
  * 
  * You should have received a copy of the GNU General Public License
  * along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

/*
  * CUDA/OpenCL capable device query program, uses the CUDA runtime API.
  * This version can be used to query *all* CUDA devices available across multiple
  * nodes using MPI. Note: it must run with a suitable 'mpiexec' or 'mpirun' command
  * to allow it to "see" all of the CUDA devices available.
  *
  * Usage: device-query 
  *
  * charliep	13-April-2011	First pass, based on deviceQuery from NVIDIA.
  * charliep	01-July-2011	Improved error handling, additional characteristics.
*/
#include <mpi.h>
#include "device-query-functions.h"
#include <stdbool.h>

#define mpiForAll(r, rank, rankCount) for (r = 0; r < rankCount; r++, MPI_Barrier(MPI_COMM_WORLD)) if (rank == r)

int main(int argc, char** argv) {
	int deviceID, deviceCount;
	
	int i, j, r;
	cudaError_t status = (cudaError_t)0;
	
	int rank, rankCount;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &rankCount);

	char name[MPI_MAX_PROCESSOR_NAME]; int name_len;
	MPI_Get_processor_name(name, &name_len);
	
	char * nameBuffer = (char*)malloc(rankCount * sizeof(name));
	MPI_Allgather(name, MPI_MAX_PROCESSOR_NAME, MPI_CHAR,
		nameBuffer, MPI_MAX_PROCESSOR_NAME, MPI_CHAR, MPI_COMM_WORLD);
	//need a bunch of pointers to each name so we can loop through them as C-strings
	char ** allNames = (char**)malloc(rankCount * sizeof(char*));
	for (i = 0; i < rankCount; i++) {
		allNames[i] = nameBuffer + i * MPI_MAX_PROCESSOR_NAME;
	}
	
	//remove duplicates
	/*int numUnique = rankCount;
	for (i = 0; i < numUnique; i++) {
		for (j = i; j < rankCount; j++) {
			//if the same
			if (strncmp(allNames[i], allNames[j], MPI_MAX_PROCESSOR_NAME) == 0) {
				//shift everything down
				for (k = j+1; k < numUnique; k++) {
					strncpy(allNames[k-1], allNames[k], MPI_MAX_PROCESSOR_NAME);
				}
				numUnique--;
			}
		}
	}*/
	//find out if yours appears earlier in list
	bool unique = true;
	for (i = 0; i < rank && unique; i++) {
		if (strncmp(allNames[i], name, MPI_MAX_PROCESSOR_NAME) == 0) {
			unique = false;
		}
	}
	
	int totalCudaDevices = 0;
	
	mpiForAll(i, rank, rankCount) {
		if (unique) {
			printf("[[ Node %s ]]\n", name);
			printf("Contains ranks:");
			for (j = 0; j < rankCount; j++) {
				if (strncmp(allNames[i], allNames[j], MPI_MAX_PROCESSOR_NAME) == 0) {
					printf(" <%d>", j);
				}
			}
			printf("\n");
	
			int driverVersion;

			printCudaVersion(&driverVersion, NULL);

			if (driverVersion == 0) {
				printf("No CUDA drivers detected--assuming no local CUDA cards.\n");
			} else {			
				deviceCount = printDeviceCount();
				totalCudaDevices += deviceCount;
				int device;
				for (device = 0; device < deviceCount; ++device) {
					printDeviceProperties(device);
				}
			}
			
			printf("------------\n");
		}
	}
	if (rank == 0) {
		printf("Total number of CUDA devices available to this job: %d\n", totalCudaDevices);
	}
	
	MPI_Finalize();
	return 0;
}


