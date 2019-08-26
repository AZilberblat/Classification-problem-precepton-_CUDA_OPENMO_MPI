#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <omp.h>
#include <stdlib.h>
#include "Constants.h"

/* incorrectLablesKernel executes the one iteration of the binary classifier algorithm
through all the points , given some weights and computes total incorrect lables

One thread is launched on the GPU corresponding to the each data point

Each threads loads coordinates of its data points to the shared memory of the GPU

and uses those to evaluate the weight function and increment the incorrect lables count

if misclassification is found */


__global__ void incorrectLablesKernel(const float*device_pts, const int*device_pts_cls,
	const float *device_weights, unsigned int *device_nmis_clsf,
	const int num_pts, const int num_coord)
{
	//ThreadId in the the block
	int thId = threadIdx.x;

	// index of the point to be computed by current thread
	int i = blockDim.x * blockIdx.x + thId;

	//Allocating dynamic shared memory for each data points in the block
	extern __shared__ float shared_points[];

	//Loading coordinates of the point in the shared memory
	if (i < num_pts)
	{
		for (int j = 0; j<num_coord; j++)
		{
			shared_points[thId*num_coord + j] = device_pts[i*num_coord + j];
		}
	}

	__syncthreads();

	if (i < num_pts)
	{
		//Evaluate discriminant function
		float f_evaluation = 0.0f;
		int j = 0;

		// Adding bias
		f_evaluation += device_weights[0];
		for (j = 1; j <= num_coord; j++)
		{
			f_evaluation += shared_points[thId*num_coord + (j - 1)] * device_weights[j];
		}

		// Predict class of the point

		int pt_cls = 1;
		if (f_evaluation < 0)
			pt_cls = -1;


		//For the count of misclassifications, atomic operations is used 
		if (device_pts_cls[i] != pt_cls)
		{
			atomicAdd(&device_nmis_clsf[0], 1);
		}

	}
}

float cudaCalc()
{
	FILE *fPtr;


	fPtr = fopen(FILE_DATA1, "r");


	if (!fPtr)
	{
		printf("File not found!\n");
		return 0;
	}

	// Obtain the relevant data for the classifier algorithm

	int num_pts, num_coord, limit;
	float alp0, alpMax, desiredQuality;
	fscanf(fPtr, "%d %d %f %f %d %f\n", &num_pts, &num_coord, &alp0, &alpMax,
		&limit, &desiredQuality);
	printf("\n==========================\n");
	printf("Parallel Implementation");
	printf("\n==========================\n");
	printf("Number of Points = %d\n", num_pts);
	printf("Number of Coordinates = %d\n", num_coord);
	printf("Maximum number of iterations = %d\n", limit);
	printf("Desired quality = %f\n", desiredQuality);
	printf("Alpha 0 = %f\n", alp0);
	printf("Alpha Max = %f\n", alpMax);


	// Calculate and assign memory for saving the points coordinates and their corresponding 
	// classes

	const unsigned int bytesForPts = num_pts * num_coord * sizeof(float);
	const unsigned int bytesForClasses = num_pts * sizeof(int);


	float *host_pts;
	int *host_pts_cls;
	cudaMallocHost(&host_pts, bytesForPts);
	cudaMallocHost(&host_pts_cls, bytesForClasses);

	// Obtain points from the input file
	for (int i = 0; i < num_pts; i++)
	{
		for (int j = 0; j<num_coord; j++)
		{
			fscanf(fPtr, "%f ", &host_pts[i*num_coord + j]);
		}
		fscanf(fPtr, "%d\n", &host_pts_cls[i]);
	}

	fclose(fPtr);

	float time_init = omp_get_wtime();


	//Allocating memory at the GPU and copying the data on the GPU
	float *device_pts;
	int *device_pts_cls;

	cudaMalloc(&device_pts, bytesForPts);
	cudaMalloc(&device_pts_cls, bytesForClasses);

	cudaMemcpy(device_pts, host_pts, bytesForPts, cudaMemcpyHostToDevice);
	cudaMemcpy(device_pts_cls, host_pts_cls, bytesForClasses, cudaMemcpyHostToDevice);


	//Computing block size and grid size 
	const int blockSize = 128;
	int gridSize = num_pts / blockSize;

	if (num_pts % blockSize != 0)
		gridSize++;

	//Allocating memory for the wights and bias at the CPU and GPU
	const unsigned int weightsBytes = (num_coord + 1) * sizeof(float);

	float *host_w = (float*)malloc(weightsBytes);
	float *host_w_new = (float*)malloc(weightsBytes);

	//TO save count of the numbers of misclassifications
	unsigned int host_nmis_clsf = 0.0f;

	for (int j = 0; j <= num_coord; j++)
	{
		host_w[j] = 0.0f;
	}

	float *device_w;
	unsigned int *device_nmis_clsf;
	cudaMalloc(&device_w, weightsBytes);
	cudaMalloc(&device_nmis_clsf, sizeof(unsigned int));



	float alp = 0.0f;
	float achieved_quality = desiredQuality + 1;

	// Iterate until desired quality is not achieved or alpha is less than
	// pred-defined 
	while (achieved_quality >= desiredQuality && alp + alp0 < alpMax + 0.1)
	{
		alp += alp0;

		// Iterate untill limit is reached
		// or all examples are correclty classified/number of misclassifications are zero
		for (int c = 0; c<limit&& achieved_quality != 0.0f; c++)
		{

			// Setting redifinition index to larger value
			// It contains the information that for which point weight are redefined
			int redifinition_idx = num_pts;

			/* points are evaluted by the threads similar to as they are distributed in
			cyclic fashion

			The first point that does not satisfies this criterion will cause to stop
			the check and immediate redefinition of the vector W, and also update the
			redefinition index

			All threads stop if they evaluated all the points below the redefinition index
			or if they found a smaller redefinition index, they will redefine the weights
			*/
#pragma omp parallel
			{
				int start_index = omp_get_thread_num();
				int increment = omp_get_num_threads();
				for (int i = start_index; i < redifinition_idx && i < num_pts; i += increment)
				{
					//Evaluate discriminant function
					float f_evaluation = 0.0f;

					// Adding bias
					f_evaluation += host_w[0];
					for (int j = 1; j <= num_coord; j++)
					{
						f_evaluation += host_pts[i*num_coord + (j - 1)] * host_w[j];
					}

					// Predict class of the point

					int pt_cls = 1;
					if (f_evaluation < 0)
						pt_cls = -1;



					//Similarly for the count of misclassifications, atomic operations also be used
					// To avoid race condition to the weights redefinition or redefinition index
					// all computation below should be performed atomically
#pragma omp critical
					{
						if (host_pts_cls[i] != pt_cls && i < redifinition_idx)
						{
							redifinition_idx = i;
							float partialResult = alp * -1 * pt_cls;
							host_w_new[0] = host_w[0] + partialResult;
							for (int j = 1; j <= num_coord; j++)
							{
								host_w_new[j] = host_w[j] + partialResult * host_pts[i*num_coord + (j - 1)];
							}
						}
					}
				}
			}

			for (int j = 0; j <= num_coord; j++)
			{
				host_w[j] = host_w_new[j];
			}

			//Moving new wights to the GPU memory
			cudaMemcpy(device_w, host_w, weightsBytes, cudaMemcpyHostToDevice);

			// Initializng number of the misclassifications to zero
			cudaMemset(device_nmis_clsf, 0, sizeof(int));

			// Calculating shared memory to be allocated per block to save the data points
			const int sharedMemoryPerThreadBlock = blockSize * num_coord * sizeof(float);

			// Execute one iteration
			incorrectLablesKernel << <gridSize, blockSize, sharedMemoryPerThreadBlock >> >
				(device_pts, device_pts_cls, device_w, device_nmis_clsf, num_pts, num_coord);

			cudaMemcpy(&host_nmis_clsf, device_nmis_clsf, sizeof(unsigned int),
				cudaMemcpyDeviceToHost);

			//Computing achieved quality after one completed iteration
			achieved_quality = (float)host_nmis_clsf / num_pts;

		}

	}

	float timeTaken = omp_get_wtime() - time_init;



	//Writing parameters to the file
	fPtr = fopen(PATH_OUTPUTP, "w");
	printf("\n------- Output ------\n\n");
	printf("Alpha = %f\n", alp);
	fprintf(fPtr, "Alpha = %f\n", alp);
	printf("Achieved quality = %f\n", achieved_quality);
	fprintf(fPtr, "Achieved quality = %f\n", achieved_quality);

	for (int j = 0; j <= num_coord; j++)
	{
		printf("w[%d] = %f\n", j, host_w[j]);
		fprintf(fPtr, "w[%d] = %f\n", j, host_w[j]);
	}

	fclose(fPtr);

	printf("(Parallel) Time elapsed = %f secs\n==========================\n", timeTaken);


	//Freeing all the memory
	free(host_w);
	free(host_w_new);
	cudaFree(device_w);
	cudaFreeHost(host_pts);
	cudaFree(device_pts);
	cudaFreeHost(host_pts_cls);
	cudaFree(device_pts_cls);
	cudaFree(device_nmis_clsf);

	return timeTaken;
}