#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <omp.h>
#include <stdlib.h>
#include "Constants.h"

// Preception algorithm Sequential Implementation

float seq()
{
	// Init vars 
	int  numprocs, myid;
	float seqTime = 0, paraTime = 0;



	// Preception algorithm Sequential Implementation

	float alpha0, alphaMax, quality_of_classifier;
	int limit, num_points = 0, num_cdts = 0;

	FILE *fPtr;


	fPtr = fopen(FILE_DATA1, "r");
	fscanf(fPtr, "%d %d %f %f %d %f", &num_points, &num_cdts, &alpha0, &alphaMax, &limit, &quality_of_classifier);

	if ((num_points < MIN_NUM_POINTS || num_points > MAX_NUM_POINTS) || (num_cdts >= MAX_DIM) || (alphaMax / alpha0 >= MAX_ALPHA_DELTA) || (limit > MAX_LIMIT))
	{
		printf("There is an exception with your file numbers, please fix it and try again...\n");
		exit(0);
	}
	printf("\n==========================\n");
	printf("Sequential Implementation");
	printf("\n==========================\n");
	printf("Number of Points = %d\n", num_points);
	printf("Number of Coordinates = %d\n", num_cdts);
	printf("Maximum number of iterations = %d\n", limit);
	printf("Desired quality = %f\n", quality_of_classifier);
	printf("Alpha 0 = %f\n", alpha0);
	printf("Alpha Max = %f\n", alphaMax);
	
	
	float *points = (float *)malloc(num_points * num_cdts * sizeof(float));
	int *labels = (int *)malloc(num_points * sizeof(int));


	int i = 0;
	while (!feof(fPtr))
	{
		for (int j = 0; j < num_cdts; j++)
		{

			fscanf(fPtr, "%f", &points[i*num_cdts + j]);

		}
		fscanf(fPtr, "%d", &labels[i]);
		i++;
	}
	fclose(fPtr);

	float start_time = omp_get_wtime();

	// initialize weights and bias parameters to all zeros
	float *weights = (float *)malloc(num_cdts * sizeof(float));
	float bias = 0.0f;

	for (int i = 0; i<num_cdts; i++)
		weights[i] = 0.0f;

	float q = quality_of_classifier + 1;

	float alpha = alpha0;
	while (q >= quality_of_classifier  && alpha <= alphaMax)
	{
		for (int z = 0; z<limit; z++)
		{
			int nmis = 0;
			for (int i = 0; i<num_points; i++)
			{
				float summation = 0.0;

				for (int j = 0; j<num_cdts; j++)
				{
					summation += (weights[j] * points[i*num_cdts + j]);
				}

				summation += bias;

				int prediction = 1;

				if (summation < 0.0f)
				{
					prediction = -1;
				}

				int actual = labels[i];
				if (actual != prediction)
				{
					float partial_product = alpha * (prediction*-1);
					for (int j = 0; j<num_cdts; j++)
					{
						weights[j] += (partial_product * points[i*num_cdts + j]);
					}
					bias += partial_product;
					break;
				}
			}

			for (int i = 0; i<num_points; i++)
			{
				float summation = 0.0;

				for (int j = 0; j<num_cdts; j++)
				{
					summation += (weights[j] * points[i*num_cdts + j]);
				}

				summation += bias;

				int prediction = 1;

				if (summation < 0.0f)
				{
					prediction = -1;
				}


				int actual = labels[i];
				if (actual != prediction)
				{
					nmis++;
				}
			}

			q = nmis / (float)num_points;

			if (nmis == 0)
				break;

		}

		alpha += alpha0;
	}

	alpha -= alpha0;

	seqTime = omp_get_wtime() - start_time;
	printf("\n------- Output ------\n\n");

	printf("Alpha = %f\n", alpha);

	printf("Achieved quality = %f\n", q);

	printf("w[0] = %f\n", bias);

	for (int i = 0; i<num_cdts; i++)
	{
		printf("w[%i] = %f\n", i + 1, weights[i]);
	}


	fPtr = fopen(PATH_OUTPUTS, "w");
	fprintf(fPtr, "Alpha = %f\nAchieved quality = %f\n", alpha, q);
	fprintf(fPtr, "w[0] = %f\n", bias);
	for (int j = 0; j<num_cdts; j++)
	{
		fprintf(fPtr, "w[%d] = %f\n", j+1 , weights[j]);
	}
	fclose(fPtr);

	free(weights);
	free(points);
	free(labels);
	printf("(Sequential) Time elapsed = %g secs\n==========================\n", seqTime);
	return seqTime;

}




