//一个简单的CUDA测试程序
#include <stdio.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <time.h>


//主函数
int main()
{
	int numElements = 1024 * 1024;  //数组大小
	size_t size = numElements * sizeof(float);
	printf("[Vector addition of %d elements]\n", numElements);

	// 数组分配host内存
	float *h_A = (float *)malloc(size);
	float *h_B = (float *)malloc(size);
	float *h_C = (float *)malloc(size);

	// 确认内存分配成功
	if (h_A == NULL || h_B == NULL || h_C == NULL)
	{
		fprintf(stderr, "Failed to allocate host vectors!\n");
		exit(EXIT_FAILURE);
	}

	//随机初始化A B
	for (int i = 0; i < numElements; ++i)
	{
		h_A[i] = rand() / (float)RAND_MAX;
		h_B[i] = rand() / (float)RAND_MAX;
	}


	//用cpu计算数据结果并计算耗时
	clock_t begin, end;
	//float *hh_C = (float *)malloc(size);
	begin = clock();
	for (int i = 0; i < numElements; ++i)
	{
		h_C[i] = h_A[i] + h_B[i];
	}
	end = clock();
	printf("cpu execution time (ms): %f \n", ((float)(end - begin)) / CLOCKS_PER_SEC);

	//释放host内存
	free(h_A);
	free(h_B);
	free(h_C);

	printf("Done\n");
	return 0;
}