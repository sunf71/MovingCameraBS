#include "MBD.h"

void FastMBD(const Mat& img, Mat& U, Mat& L, int k, Mat& seeds, Mat& mbdMap)
{
	mbdMap.create(img.size(), CV_32F);
	mbdMap.setTo(Scalar(MAXD), seeds);
	mbdMap.setTo(Scalar(0), ~seeds);
	L = img.clone();
	U = img.clone();
	for (size_t i = 0; i < k; i++)
	{
		if (i % 2 == 0)
			RasterScan(img, mbdMap, U, L, true);
		else
			RasterScan(img, mbdMap, U, L, false);
	}
}

void RasterScan(const Mat& img, Mat& mbdMap, Mat& U, Mat& L, bool order)
{
	//set scan order
	size_t rs(0), re(img.rows-1);
	size_t cs(0), ce(img.cols-1);
	if (order)
	{
		for (size_t i = rs; i <= re; i++)
		{
			for (size_t j = cs; j <= ce; j++)
			{

			}

		}
	}
	else
	{
		for (size_t i = re; i >= rs; i--)
		{
			for (size_t j = ce; j >= cs; j--)
			{

			}

		}
	}
	
}