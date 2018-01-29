#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <string>
#include <vector>
#include <map>
#include <iostream>
#include <sstream>
#include "device_functions.h"
#include <helper_cuda.h>
#include <fstream>
using std::vector;
using std::string;
using std::map;

#define THREAD_NUM 512
#define BLOCK_NUM 1024
const float pi = 3.1415926;

__constant__ static int dd_angle[30];
__constant__ static float dd_start[30];
__constant__ static float d_s_angle_1[30];
__constant__ static float d_s_angle_2[30];


struct InitData   //换能器基本数据
{
	InitData(int c, int fs, int f0, float width, float kerf, int N_elements, int length, float image_length, vector<int> angle) :
		speed(c), sample_frequency(fs), central_frequency(f0), width(width), kerf(kerf), N_elements(N_elements), data_length(length), image_length(image_length), angle(angle)
	{
		pitch = width + kerf;
		array_length = pitch*(N_elements - 1) + width;
		d_x = array_length / N_elements;
		d_z = double(1) / fs;
	}

	void push_tstart(float tstart, int i)
	{
		tstatrt[i] = tstart;
	}

	//原始参数
	int speed;
	float sample_frequency;
	int central_frequency;
	float width;
	float kerf;
	int N_elements;
	int data_length;
	float pitch;
	float array_length;
	float d_x;
	float image_length;
	double d_z;
	vector<int> angle;
	map<int, float> tstatrt;
};

// analyse和readData是读取dat文件的函数
void analyse(float* in, const char* buf)
{
	string contents = buf;
	string::size_type pos1 = 0;
	int n = 0;
	int i = 0;
	while ((pos1 = contents.find_first_of("+-.0123456789e", pos1)) != string::npos)
	{
		auto pos2 = contents.find_first_not_of("+-.0123456789e", pos1);
		n = pos2 - pos1;
		float d = stod(contents.substr(pos1, n));
		in[i++] = d;
		pos1 += n;
	}
}

float* readData(string path, InitData &init)
{
	int one_frame_length = init.N_elements*init.data_length;
	int all_data_length = (init.angle.size())*one_frame_length;
	float *all_rf = new float[all_data_length];
	float *t_start = new float[init.angle.size()];

	const int MAXS = one_frame_length * 20;//数字字符数量
	char *buf = new char[MAXS];
	char *t_buf = new char[20];
	int kk = 0;

	for (auto ii : init.angle)
	{
		std::cout << "正在读取第" << ii << "帧数据" << std::endl;

		std::stringstream pathname;
		pathname << ii;
		string file_path_now = path + "data_" + pathname.str() + ".dat";
		std::ifstream ifs(file_path_now, std::ios::binary);
		if (ifs)
		{
			float *data = all_rf + one_frame_length*kk;
			ifs.read((char*)data, one_frame_length*sizeof(data));

		}

		string t_path = path + "tstart_" + pathname.str() + ".txt";
		const char* t_file_path = t_path.c_str();
		FILE* t_fp = fopen(t_file_path, "rb");

		if (t_fp)
		{
			int len = fread(t_buf, 1, 20, t_fp);
			t_buf[len] = '\0';
			analyse(t_start + kk, t_buf);//连续存入
			init.push_tstart(t_start[kk], ii);
		}
		kk++;
	}
	delete buf;
	return all_rf;

}

//第一步并行计算的核函数
__global__ void cuda_compoundData(float* out, float* in, int new_length, int length, int N_elements, float pitch, int angle_n,
	int fs, int c, double d_z)
{
	int col = blockIdx.x*blockDim.x + threadIdx.x;
	int row = blockIdx.y*blockDim.y + threadIdx.y;

	int angle_index = row / N_elements;
	float t1 = dd_start[angle_index];
	float temp1 = d_s_angle_1[angle_index];
	float temp2 = d_s_angle_2[angle_index];
	int real_row = row - angle_index*N_elements;
	float i_real_own = (dd_angle[angle_index] > 0) ? real_row*pitch : (N_elements - real_row - 1)*pitch;

	float j_real = d_z*(col + 1) *c / 2;

	//减少下面循环里面的计算量
	float j_real_2 = j_real*j_real;
	int oneFrameLength = N_elements*new_length;
	float j_temp1 = j_real*temp1;
	float i_temp2 = i_real_own*temp2;
	for (int row_i = 0; row_i != N_elements; ++row_i)
	{
		float i_real = (real_row - row_i)*pitch;
		int jj = ((j_temp1 + i_temp2 + (sqrtf(j_real_2 + i_real *i_real))) / c - t1)*fs - 0.5f;//确定数据所在索引
		if ((jj >= 0) && (jj < new_length))
		{
			out[row*new_length + col] += in[angle_index*oneFrameLength + row_i*new_length + jj];
		}
	}
}

//第二步并行计算的核函数
__global__ void cuda_AddData(float* out, float* in, int length, int N_elements, int angle_n)
{
	int col = blockIdx.x*blockDim.x + threadIdx.x;
	int row = blockIdx.y*blockDim.y + threadIdx.y;

	for (int i = 0; i != angle_n; ++i)
	{
		out[row*length + col] += in[i*N_elements*length + row*length + col];
	}
}

//调用核函数的主要计算函数
void compoundData(float* data, InitData &init)
{
	//数据量的计算
	int one_frame_length = init.N_elements*init.data_length;
	int all_data_length = (init.angle.size())*one_frame_length;

	int new_length = (init.data_length / 32 + 1) * 32;//线程束的倍数
													  //int new_length = 64;
	int new_one_frame_length = init.N_elements*new_length;
	int new_all_data_length = (init.angle.size())*new_one_frame_length;


	std::cout << "最大数据长度为" << new_length;

	//计时函数
	cudaEvent_t startMemcpy; cudaEvent_t stopMemcpy;
	cudaEvent_t startKernel; cudaEvent_t stopKernel;

	cudaEventCreate(&startMemcpy);
	cudaEventCreate(&stopMemcpy);
	cudaEventCreate(&startKernel);
	cudaEventCreate(&stopKernel);

	cudaEventRecord(startMemcpy);  //计算GPU中复制和开辟数据所需时间

	float *new_data = new float[new_all_data_length]();//结果数据 主机和设备内存的开辟
	for (int kk = 0; kk != init.angle.size(); ++kk)
	{
		for (int jj = 0; jj < init.data_length; jj++) {
			for (int ii = 0; ii < init.N_elements; ii++) {
				new_data[kk*new_one_frame_length + ii*new_length + jj] = data[kk*one_frame_length + ii*init.data_length + jj];
			}
		}
	}

	//开辟device内存
	float *d_new_data;
	cudaMalloc(&d_new_data, new_all_data_length*sizeof(float));
	cudaMemcpy(d_new_data, new_data, sizeof(float) * new_all_data_length, cudaMemcpyHostToDevice);

	float *d_ans_data;                                    //计算数据 设备内存的开辟和赋值
	cudaMalloc(&d_ans_data, new_one_frame_length*sizeof(float));
	float *ans_data = new float[new_one_frame_length]();


	const size_t smemSize = THREAD_NUM*sizeof(float);

	//常量内存 存放角度等数据
	int *d_angle = new int[init.angle.size()];
	float *d_start = new float[init.angle.size()];
	float *s_angle_1 = new float[init.angle.size()];
	float *s_angle_2 = new float[init.angle.size()];
	for (int i = 0; i != init.angle.size(); ++i)
	{
		d_angle[i] = init.angle[i];
		d_start[i] = init.tstatrt[init.angle[i]];
		s_angle_1[i] = cos(float(d_angle[i])*pi / 180);
		s_angle_2[i] = sin(float(d_angle[i])*pi / 180);
	}

	//设置常量内存
	cudaMemcpyToSymbol(dd_angle, d_angle, sizeof(int) * init.angle.size());
	cudaMemcpyToSymbol(dd_start, d_start, sizeof(float) * init.angle.size());
	cudaMemcpyToSymbol(d_s_angle_1, s_angle_1, sizeof(float) * init.angle.size());
	cudaMemcpyToSymbol(d_s_angle_2, s_angle_2, sizeof(float) * init.angle.size());

	checkCudaErrors(cudaPeekAtLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	cudaEventRecord(stopMemcpy);

	cudaEventRecord(startKernel);//计算核函数用时

	float *d_temp_data;                                    //计算数据 设备内存的开辟和赋值
	cudaMalloc(&d_temp_data, new_all_data_length*sizeof(float));
	float *temp_data = new float[new_all_data_length]();
	
	//调用第一个核函数
	dim3 dimBlock(8, 8, 1);
	dim3 dimGrid((new_length + dimBlock.x - 1) / dimBlock.x,
		(init.N_elements*init.angle.size() + dimBlock.y - 1) / dimBlock.y, 1);

	cuda_compoundData << <dimGrid, dimBlock >> >(d_temp_data, d_new_data, new_length, init.data_length, init.N_elements, init.pitch, init.angle.size(),
		init.sample_frequency, init.speed, init.d_z);
	checkCudaErrors(cudaPeekAtLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	//调用第二个核函数
	dim3 dimBlock2(8, 8, 1);
	dim3 dimGrid2((new_length + dimBlock.x - 1) / dimBlock.x,
		(init.N_elements + dimBlock.y - 1) / dimBlock.y, 1);

	cuda_AddData << <dimGrid2, dimBlock2 >> >(d_ans_data, d_temp_data, new_length, init.N_elements, init.angle.size());
	cudaEventRecord(stopKernel);

	checkCudaErrors(cudaPeekAtLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	//复制device结果至host
	cudaMemcpy(temp_data, d_temp_data, sizeof(float) * new_all_data_length, cudaMemcpyDeviceToHost);
	cudaMemcpy(ans_data, d_ans_data, sizeof(float) * new_one_frame_length, cudaMemcpyDeviceToHost);

	//销毁host和device内存
	cudaFree(dd_angle);
	cudaFree(dd_start);
	cudaFree(d_s_angle_1);
	cudaFree(d_s_angle_2);
	cudaFree(d_ans_data);
	cudaFree(d_new_data);

	delete d_angle;
	delete d_start;
	delete s_angle_1;
	delete s_angle_2;
	delete data;
	delete new_data;
	delete temp_data;
	delete ans_data;

	//计算用时
	float memcpyTime = 0;
	cudaEventElapsedTime(&memcpyTime, startMemcpy, stopMemcpy);
	float kernelTime = 0;
	cudaEventElapsedTime(&kernelTime, startKernel, stopKernel);

	std::cout << "GPU和CPU中复制数据用时" << memcpyTime << "ms" << std::endl;
	std::cout << "核函数计算用时（多角度复合）" << kernelTime << "ms" << std::endl;

}

int main()
{
	string path = "..//data//";
	vector<int> angle = { -9,-7,-5,-3,-1, 0,1,3,5,7,9 };
	//vector<int> angle = { -7,-5,-3,-1, 0,1,3,5,7 };

	InitData init(1540, 50e6, 3.5e6, 0.2798e-3, 0.025e-3, 128, 6000, 0.11, angle);

	//std::cout << cos(1*pi/180);
	float* test = readData(path, init);

	compoundData(test, init);
	//std::cout << *test;
	std::cout << "计算结束！";
}