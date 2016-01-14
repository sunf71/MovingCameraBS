#include "MBD.h"

void FastMBD(const Mat& img, Mat& U, Mat& L, int k, Mat& seeds, Mat& mbdMap)
{
	mbdMap.create(img.size(), CV_32F);
	mbdMap.setTo(Scalar(MAXD), seeds);
	mbdMap.setTo(Scalar(0), ~seeds);
	L = img.clone();
	L.convertTo(L, CV_32F);
	U = img.clone();
	U.convertTo(U, CV_32F);
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
	int width = img.cols;
	int height = img.rows;
	//set scan order
	size_t rs(0), re(img.rows-1);
	size_t cs(0), ce(img.cols-1);
	if (order)
	{
		for (size_t i = rs; i <= re; i++)
		{
			for (size_t j = cs; j <= ce; j++)
			{
				int nr = i - 1;
				float lx = img.at<uchar>(i, j)*1.f;
				if (nr >= 0)
				{
					int idx_n = nr*width + j;
					float Uy = U.at<float>(nr, j);
					float Ly = L.at<float>(nr, j);
					float B = std::max(Uy, lx) - std::min(Ly,lx);
					if (mbdMap.at<float>(i, j) < B)
					{
						mbdMap.at<float>(i, j) = B;
						U.at<float>(i, j) = std::max(Uy, lx);
						L.at<float>(i, j) = std::min(Ly, lx);
					}
				}
				int nc = j - 1;
				if (nc >= 0)
				{
					
					float Uy = U.at<float>(i, nc);
					float Ly = L.at<float>(i, nc);
					float B = std::max(Uy, lx) - std::min(Ly, lx);
					if (mbdMap.at<float>(i, j) < B)
					{
						mbdMap.at<float>(i, j) = B;
						U.at<float>(i, j) = std::max(Uy, lx);
						L.at<float>(i, j) = std::min(Ly, lx);
					}
				}
			}

		}
	}
	else
	{
		for (size_t i = re; i >= rs; i--)
		{
			for (size_t j = ce; j >= cs; j--)
			{
				int nr = i - 1;
				float lx = img.at<uchar>(i, j)*1.f;
				if (nr >= 0)
				{
					int idx_n = nr*width + j;
					float Uy = U.at<float>(nr, j);
					float Ly = L.at<float>(nr, j);
					float B = std::max(Uy, lx) - std::min(Ly, lx);
					if (mbdMap.at<float>(i, j) < B)
					{
						mbdMap.at<float>(i, j) = B;
						U.at<float>(i, j) = std::max(Uy, lx);
						L.at<float>(i, j) = std::min(Ly, lx);
					}
				}
				int nc = j - 1;
				if (nc >= 0)
				{

					float Uy = U.at<float>(i, nc);
					float Ly = L.at<float>(i, nc);
					float B = std::max(Uy, lx) - std::min(Ly, lx);
					if (mbdMap.at<float>(i, j) < B)
					{
						mbdMap.at<float>(i, j) = B;
						U.at<float>(i, j) = std::max(Uy, lx);
						L.at<float>(i, j) = std::min(Ly, lx);
					}
				}
			}

		}
	}
	
}