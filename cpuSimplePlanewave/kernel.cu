#define _CRT_SECURE_NO_WARNINGS

#include <stdio.h>
#include <string>
#include <vector>
#include <map>
#include <iostream>
#include <sstream>
#include <time.h>
#include <fstream>
using std::vector;
using std::string;
using std::map;

#define THREAD_NUM 512
#define BLOCK_NUM 1024
const float pi = 3.1415926;



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

void analyse(float* in, const char* buf)
{
	string contents = buf;
	string::size_type pos1 = 0;
	int n = 0;
	int i = 0;
	while ((pos1 = contents.find_first_of("+-.0123456789e", pos1)) != string::npos)//经测试，较快的数据读取方法（一次性读入string再分析）
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



void compoundData(float* data, InitData &init)
{
	int one_frame_length = init.N_elements*init.data_length;
	int all_data_length = (init.angle.size())*one_frame_length;

	int new_length = (init.data_length / 32 + 1) * 32;//线程束的倍数
													  //int new_length = 64;
	int new_one_frame_length = init.N_elements*new_length;
	int new_all_data_length = (init.angle.size())*new_one_frame_length;


	std::cout << "最大数据长度为" << new_length << std::endl;

	float *new_data = new float[new_all_data_length]();//结果数据 主机和设备内存的开辟
	for (int kk = 0; kk != init.angle.size(); ++kk)
	{
		for (int jj = 0; jj < init.data_length; jj++) {
			for (int ii = 0; ii < init.N_elements; ii++) {
				new_data[kk*new_one_frame_length + ii*new_length + jj] = data[kk*one_frame_length + ii*init.data_length + jj];
			}
		}
	}


	float *ans_data = new float[new_one_frame_length]();


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


	int ang_size = init.angle.size();
	int N_elements = init.N_elements;
	float pitch = init.pitch;
	float d_z = init.d_z;
	int c = init.speed;
	int fs = init.sample_frequency;

	clock_t begin, end;
	begin = clock();

	//设置四层循环计算索引
	for (int angle_index = 0; angle_index != ang_size; ++angle_index)
	{
		std::cout << "计算第" << angle_index + 1 << "个角度" << std::endl;
		float t1 = d_start[angle_index];
		float temp1 = s_angle_1[angle_index];
		float temp2 = s_angle_2[angle_index];

		for (int row = 0; row != N_elements; ++row)
		{
			float i_real_own = (d_angle[angle_index] > 0) ? row*pitch : (N_elements - row - 1)*pitch;

			for (int col = 0; col != new_length; ++col)
			{
				float j_real = d_z*(col + 1) *c / 2;
				for (int kk = 0; kk != N_elements; ++kk)
				{
					int row_i = kk - angle_index*N_elements;
					float i_real = (row - row_i)*pitch;
					int jj = ((j_real*temp1 + i_real_own*temp2 + (sqrtf(j_real*j_real + i_real *i_real))) / c - t1)*fs - 0.5f;
					if ((jj >= 0) && (jj < new_length))
					{
						ans_data[row*new_length + col] += new_data[angle_index*new_one_frame_length + row_i*new_length + jj];
					}
				}
			}
		}
	}

	end = clock();

	delete d_angle;
	delete d_start;
	delete s_angle_1;
	delete s_angle_2;
	delete data;
	delete new_data;
	delete ans_data;

	std::cout << "CPU计算用时（多角度复合）" << double((end - begin) / CLOCKS_PER_SEC) * 1000 << "ms" << CLOCKS_PER_SEC << std::endl;

}

int main()
{
	string path = "..//data//";
	vector<int> angle = { -9,-7,-5,-3,-1, 0,1,3,5,7,9 };

	InitData init(1540, 50e6, 3.5e6, 0.2798e-3, 0.025e-3, 128, 6000, 0.11, angle);

	float* test = readData(path, init);

	compoundData(test, init);
	//std::cout << *test;
	std::cout << "计算结束！";
}