#include "APAPWarping.h"
#include "findHomography.h"

double pdist2(double x1, double y1, double x2, double y2)
{
	double dx = x1 - x2;
	double dy = y1 - y2;
	return dx*dx + dy*dy;
}
cv::Mat  rollVector9f(const double* ptr) 
{

	cv::Mat H(cv::Size(3,3), CV_64F, (void*)ptr);
	
	return H;
}
void APAPWarping::Solve()
{
	std::vector<cv::Point2f>& f1(*_x1), &f2(*_x2);

	for (int i = 0; i < _blkSize; i++) {


		cv::Mat Wi = cv::Mat::zeros(f1.size() * 2, f1.size() * 2, CV_64F);
		for (int j = 0; j < f1.size(); j++) {
			double dist_weight = exp(-pdist2(_blkCenters[i].x, _blkCenters[i].y, f1[j].x, f1[j].y) / _sigmaSquared);
			double weight = std::max(dist_weight, _gamma);
			Wi.at<double>(j * 2, j * 2) = weight;
			Wi.at<double>(j * 2 + 1, j * 2 + 1) = weight;
		}

		cv::Mat w, u, vt;
		//std::cout<<"A = "<<dataM<<std::endl;
		cv::SVDecomp(Wi*_A, w, u, vt);
		//std::cout<<"vt = "<<vt<<std::endl;
		double* ptr = (double*)(vt.data + (vt.rows - 1)*vt.step.p[0]);

		for (int j = 0; j<8; j++)
			_blkHomoVec[i*8+j] = ptr[j] / ptr[8];
		cv::Mat h = rollVector9f(ptr);
		cv::Mat invH = h.inv();
		ptr = (double*)(invH.data);
		for (int j = 0; j < 8; j++)
			_blkInvHomoVec[i * 8 + j] = ptr[j] / ptr[8];
	}


}

void APAPWarping::getFlow(cv::Mat& flow)
{

}

void APAPWarping::Warp(const cv::Mat& img, cv::Mat& warpedImg)
{
	for (int i = 0; i<_mesh.rows - 1; i++)
	{
		int * ptr0 = _mesh.ptr<int>(i);
		int * ptr1 = _mesh.ptr<int>(i + 1);
		for (int j = 0; j<_mesh.cols - 1; j++)
		{
			int idx = i*_quadStep + j;
			int x0 = ptr0[2 * j];
			int y0 = ptr0[2 * j + 1];
			cv::Point2i pt0(x0, y0);
			int x1 = ptr1[2 * (j + 1)];
			int y1 = ptr1[2 * (j + 1) + 1];
			cv::Point2i pt1(x1, y1);
			cv::Rect roi(pt0, pt1);
			cv::Mat srcQuad = _srcPtMat(roi);
			cv::Mat dstQuad = _map(roi);
			cv::Mat idstQuad = _invMap(roi);
			/*MatrixTimesMatPoints(_blkHomos[idx],srcQuad,dstQuad);
			MatrixTimesMatPoints(_blkInvHomos[idx],srcQuad,idstQuad);*/
			const double * hptr = &_blkHomoVec[idx * 8];
			const double * ihptr = &_blkInvHomoVec[idx * 8];
			for (int i = 0; i<srcQuad.rows; i++)
			{
				const float* ptr = srcQuad.ptr<float>(i);
				float* dstPtr = dstQuad.ptr<float>(i);
				float* idstPtr = idstQuad.ptr<float>(i);
				for (int j = 0; j<srcQuad.cols; j++)
				{
					float x = hptr[0] * ptr[2 * j] + hptr[1] * ptr[2 * j + 1] + hptr[2];
					float y = hptr[3] * ptr[2 * j] + hptr[4] * ptr[2 * j + 1] + hptr[5];
					float z = hptr[6] * ptr[2 * j] + hptr[7] * ptr[2 * j + 1] + 1;
					dstPtr[2 * j] = x / z;
					dstPtr[2 * j + 1] = y / z;
					x = ihptr[0] * ptr[2 * j] + ihptr[1] * ptr[2 * j + 1] + ihptr[2];
					y = ihptr[3] * ptr[2 * j] + ihptr[4] * ptr[2 * j + 1] + ihptr[5];
					z = ihptr[6] * ptr[2 * j] + ihptr[7] * ptr[2 * j + 1] + 1;
					idstPtr[2 * j] = x / z;
					idstPtr[2 * j + 1] = y / z;
				}
			}

		}
	}
	//warpedImg = img.clone();

	cv::split(_map, _mapXY);
	cv::remap(img, warpedImg, _mapXY[0], _mapXY[1], CV_INTER_CUBIC);
}
void APAPWarping::GpuWarp(const cv::gpu::GpuMat& dimg, cv::gpu::GpuMat& dwimg)
{

}
void APAPWarping::WarpPt(const cv::Point2f& input, cv::Point2f& output)
{

}
void APAPWarping::SetFeaturePoints(std::vector<cv::Point2f>& f2, std::vector<cv::Point2f>& f1)
{
	
	//// Normalise each set of points 
	//
	//std::vector<cv::Point2f> x1, x2;
	//NormalizePoints(f1, _T1, x1);
	//NormalizePoints(f2, _T2, x2);
	_x1 = &f1;
	_x2 = &f2;
	
	_A.create(2 * f1.size(), 9, CV_64F);
	double* ptr = (double*)_A.data;
	int rowStep = _A.step.p[0];
	for (int i = 0; i<f1.size(); i++)
	{
		ptr[0] = ptr[1] = ptr[2] = 0;
		ptr[3] = -1 * f1[i].x;
		ptr[4] = -1 * f1[i].y;
		ptr[5] = -1;
		ptr[6] = f2[i].y*f2[i].x;
		ptr[7] = f2[i].y*f1[i].y;
		ptr[8] = f2[i].y;
		ptr += 9;
		ptr[0] = f1[i].x;
		ptr[1] = f1[i].y;
		ptr[2] = 1;
		ptr[3] = ptr[4] = ptr[5] = 0;
		ptr[6] = -f2[i].x * f1[i].x;
		ptr[7] = -f2[i].x * f1[i].y;
		ptr[8] = -f2[i].x;
		ptr += 9;
	}
}