//一个简单的CUDA测试程序
#include <stdio.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <time.h>

//数组相加 核函数
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

//主函数
int main()
{
	int numElements = 1024*1024;  //数组大小
	int repeat_M = 1;  //计算重复次数
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

	//数组分配device内存 
	float *d_A, *d_B, *d_C;
	cudaMalloc(&d_A, size);
	cudaMalloc(&d_B, size);
	cudaMalloc(&d_C, size);

	//从host内的数组复制数据进入device内的数组
	cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

	//设置线程块和线程网格大小
	int threadsPerBlock = 128;
	int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
	printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);

	//设置计时计算
	float milli = 0;  //时间量
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	//计时开始
	cudaEventRecord(start);

	//调用核函数
	vectorAdd << <blocksPerGrid, threadsPerBlock >> >(d_A, d_B, d_C, numElements, repeat_M);

	//计时结束
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	//cudaDeviceSynchronize();
	cudaEventElapsedTime(&milli, start, stop);
	printf("gpu execution time (ms): %f \n", milli);

	//将计算结果复制回host内存内的数组
	cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

	//用cpu计算数据结果并计算耗时
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

	//与GPU计算结果进行比较 判断正确率
	for (int i = 0; i < numElements; ++i)
	{
		if (fabs(hh_C[i] - h_C[i]) > 1e-5)
		{
			fprintf(stderr, "Result verification failed at element %d!\n", i);
			exit(EXIT_FAILURE);
		}
	}
	printf("Test PASSED\n");

	//释放device内存
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);

	//释放host内存
	free(h_A);
	free(h_B);
	free(h_C);

	printf("Done\n");
	return 0;
}