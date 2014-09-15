#include "GpuSuperpixel.h"

void GpuSuperpixel::Init()
{
	cudaMalloc(&d_rgbaBuffer,m_size*4);

}

void GpuSuperpixel::Superixel(unsigned int * rgbBuffer,unsigned width, unsigned height, int step, float alpha,int& num,int* lables)
{
	m_width = width;
	m_height = height;
	m_size = width*height;
	m_step = step;
	m_alpha = alpha;
	Init();
}
