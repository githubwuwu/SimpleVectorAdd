//һ���򵥵�CUDA���Գ���
#include <stdio.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <time.h>


//������
int main()
{
	int numElements = 1024 * 1024;  //�����С
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


	//��cpu�������ݽ���������ʱ
	clock_t begin, end;
	//float *hh_C = (float *)malloc(size);
	begin = clock();
	for (int i = 0; i < numElements; ++i)
	{
		h_C[i] = h_A[i] + h_B[i];
	}
	end = clock();
	printf("cpu execution time (ms): %f \n", ((float)(end - begin)) / CLOCKS_PER_SEC);

	//�ͷ�host�ڴ�
	free(h_A);
	free(h_B);
	free(h_C);

	printf("Done\n");
	return 0;
}