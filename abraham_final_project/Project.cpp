#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <omp.h>

#include "kernel_Header.h"
#include "seq.h"
#include "Constants.h"

///////////////////////////////////
//    Perceptron Classifier      //
//     Abraham Zilberblat        //
//          308088079            //
///////////////////////////////////


int main(int argc, char **argv)
{
	// Init vars and  mpi
	int  numprocs, myid;
	float seqTime = 0, paraTime = 0;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &myid);
	MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
	MPI_Status status;

	// Preception algorithm Sequential Implementation
	if (myid == 0) {
		seqTime = seq();
	}

	// Parallel Implementation
	else {
		paraTime = cudaCalc();
		MPI_Send(&paraTime, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
	}

	if (myid == 0)
	{
		MPI_Recv(&paraTime, 1, MPI_INT, 1, 0, MPI_COMM_WORLD, &status);
		printf("\n==========================\nSequential Time is = %f\nParallel Time is = %f\n==========================\n", seqTime, paraTime);
		printf("Delta => %f\n==========================\n", (seqTime - paraTime));
	}

	MPI_Finalize();
	return 0;
}