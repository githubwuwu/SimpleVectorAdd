#include <stdio.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <time.h> 

#define N (1024*1024)  
#define M (10)  
#define THREADS_PER_BLOCK 1024  

void serial_add(double *a, double *b, double *c, int n, int m)
{
	for (int index = 0; index<n; index++)
	{

		c[index] = a[index] + b[index];
	}
}


int main()
{
	clock_t start, end;

	//double *a, *b, *c;
	int size = N * sizeof(double);

	float *h_A = (float *)malloc(size);
	float *h_B = (float *)malloc(size);
	float *h_C = (float *)malloc(size);

	for (int i = 0; i < N; ++i)
	{
		h_A[i] = rand() / (float)RAND_MAX;
		h_B[i] = rand() / (float)RAND_MAX;
	}

	start = clock();
	//serial_add(a, b, c, N, M);
	for (int index = 0; index<N; index++)
	{

		h_C[index] = h_A[index] + h_B[index];
	}

	//printf("c[%d] = %f\n", 0, c[0]);
	//printf("c[%d] = %f\n", N - 1, c[N - 1]);

	end = clock();

	float time1 = ((float)(end - start)) / CLOCKS_PER_SEC;
	printf("CPU: %f seconds\n", time1);


	free(h_A);
	free(h_B);
	free(h_C);


	return 0;
}