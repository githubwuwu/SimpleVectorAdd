//һ���򵥵�CUDA���Գ���
#include <stdio.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <time.h>

//������� �˺���
__global__ void
vectorAdd(const float *A, const float *B, float *C, int numElements,int M)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < numElements)
	{
		for (int j = 0; j < M; j++)
		{
			C[i] = A[i] * B[i];
		}
	}
}

//������
int main()
{
	int numElements = 1024*1024;  //�����С
	int repeat_M = 1;  //�����ظ�����
	size_t size = numElements * sizeof(float);
	printf("[Vector addition of %d elements]\n", numElements);

	// �������host�ڴ�
	float *h_A = (float *)malloc(size);
	float *h_B = (float *)malloc(size);
	float *h_C = (float *)malloc(size);

	// ȷ���ڴ����ɹ�
	if (h_A == NULL || h_B == NULL || h_C == NULL)
	{
		fprintf(stderr, "Failed to allocate host vectors!\n");
		exit(EXIT_FAILURE);
	}

	//�����ʼ��A B
	for (int i = 0; i < numElements; ++i)
	{
		h_A[i] = rand() / (float)RAND_MAX;
		h_B[i] = rand() / (float)RAND_MAX;
	}

	//�������device�ڴ� 
	float *d_A, *d_B, *d_C;
	cudaMalloc(&d_A, size);
	cudaMalloc(&d_B, size);
	cudaMalloc(&d_C, size);

	//��host�ڵ����鸴�����ݽ���device�ڵ�����
	cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

	//�����߳̿���߳������С
	int threadsPerBlock = 128;
	int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
	printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);

	//���ü�ʱ����
	float milli = 0;  //ʱ����
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	//��ʱ��ʼ
	cudaEventRecord(start);

	//���ú˺���
	vectorAdd << <blocksPerGrid, threadsPerBlock >> >(d_A, d_B, d_C, numElements, repeat_M);

	//��ʱ����
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	//cudaDeviceSynchronize();
	cudaEventElapsedTime(&milli, start, stop);
	printf("gpu execution time (ms): %f \n", milli);

	//�����������ƻ�host�ڴ��ڵ�����
	cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

	//��cpu�������ݽ���������ʱ
	clock_t begin, end;
	float *hh_C = (float *)malloc(size);
	begin = clock();
	for (int i = 0; i < numElements; ++i)
	{
		for (int j = 0; j < repeat_M; j++)
		{
			hh_C[i] = h_A[i] * h_B[i];
		}
	}
	end = clock();
	printf("cpu execution time (ms): %f \n", ((float)(end - begin)) / CLOCKS_PER_SEC*1000);

	//��GPU���������бȽ� �ж���ȷ��
	for (int i = 0; i < numElements; ++i)
	{
		if (fabs(hh_C[i] - h_C[i]) > 1e-5)
		{
			fprintf(stderr, "Result verification failed at element %d!\n", i);
			exit(EXIT_FAILURE);
		}
	}
	printf("Test PASSED\n");

	//�ͷ�device�ڴ�
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);

	//�ͷ�host�ڴ�
	free(h_A);
	free(h_B);
	free(h_C);

	printf("Done\n");
	return 0;
}