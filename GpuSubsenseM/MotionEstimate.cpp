#include "MotionEstimate.h"
#include "GpuSuperpixel.h"
#include <algorithm>
void postProcess(const Mat& img, Mat& mask)
{
	cv::Mat m_oFGMask_PreFlood(img.size(),CV_8U);
	cv::Mat m_oFGMask_FloodedHoles(img.size(),CV_8U);
	cv::morphologyEx(mask,m_oFGMask_PreFlood,cv::MORPH_CLOSE,cv::Mat());
	m_oFGMask_PreFlood.copyTo(m_oFGMask_FloodedHoles);
	cv::floodFill(m_oFGMask_FloodedHoles,cv::Point(0,0),UCHAR_MAX);
	cv::bitwise_not(m_oFGMask_FloodedHoles,m_oFGMask_FloodedHoles);
	cv::erode(m_oFGMask_PreFlood,m_oFGMask_PreFlood,cv::Mat(),cv::Point(-1,-1),3);
	cv::bitwise_or(mask,m_oFGMask_FloodedHoles,mask);
	cv::bitwise_or(mask,m_oFGMask_PreFlood,mask);
	cv::medianBlur(mask,mask,3);
	
}
void RegionGrowing(std::vector<cv::Point2f>& seeds,const cv::Mat& img, cv::Mat& result)
{	
	const int nx[] = {-1,0,1,0};
	const int ny[] = {0,-1,0,1};
	int width = img.cols;
	int height = img.rows;
	float3 avgColor = make_float3(0,0,0);
	float pixMaxDist(0);
	float regMaxDist = 0.2;
	int regSize(0);
	int imgSize = width*height;
	char* visited = new char[imgSize];
	std::vector<cv::Point2i> neighbors;
	float regMean(0);
	for(int i=0; i<seeds.size(); i++)
	{
		memset(visited ,0,imgSize);
		pixMaxDist = 0;
		regSize = 0;
		regMean = 0;
		neighbors.clear();
		int ix = seeds[i].x;
		int iy = seeds[i].y;
		while(pixMaxDist < regMaxDist && regSize<imgSize)
		{
			for(int d=0; d<4; d++)
			{
				int x = ix+nx[d];
				int y = iy + ny[d];
				if (!visited[x+y*width] && x>=0 && x<width && y>=0 && y<=height)
				{
					neighbors.push_back(cv::Point2i(x,y));
				}
			}
			for(
		}
	}
}
void MotionEstimate::EstimateMotion( Mat& curImg,  Mat& prevImg, Mat& transM, Mat& mask)
{
	//super pixel
	//CV_ASSERT(curImg.rows == _height && curImg.cols == _width);
	int num;
	cv::Mat img0,img1;
	cv::cvtColor(curImg,img0,CV_BGR2BGRA);
	cv::cvtColor(prevImg,img1,CV_BGR2BGRA);
	for(int i=0; i< _width; i++)
	{		
		for(int j=0; j<_height; j++)
		{
			int idx = img0.step[0]*j + img0.step[1]*i;
			_imgData0[i + j*_width].x = img0.data[idx];
			_imgData0[i + j*_width].y = img0.data[idx+ img0.elemSize1()];
			_imgData0[i + j*_width].z = img0.data[idx+2*img0.elemSize1()];
			_imgData0[i + j*_width].w = img0.data[idx+3*img0.elemSize1()];			

			_imgData1[i + j*_width].x = img1.data[idx];
			_imgData1[i + j*_width].y = img1.data[idx+ img1.elemSize1()];
			_imgData1[i + j*_width].z = img1.data[idx+2*img1.elemSize1()];
			_imgData1[i + j*_width].w = img1.data[idx+3*img1.elemSize1()];
		}
	}
	SLICClusterCenter* centers0,*centers1;
	centers0 = new SLICClusterCenter[_nSuperPixels];
	centers1 = new SLICClusterCenter[_nSuperPixels];
	_gs->Superpixel(_imgData0,num,_labels0,centers0);
	_gs->Superpixel(_imgData1,num,_labels1,centers1);
	
	//Good Features
	cv::Mat gray,preGray;
	cv::cvtColor(curImg,gray,CV_BGR2GRAY);
	cv::cvtColor(prevImg,preGray,CV_BGR2GRAY);
	cv::goodFeaturesToTrack(gray,_features0,_maxCorners,_dataQuality,_minDist);
	cv::goodFeaturesToTrack(preGray,_features1,_maxCorners,_dataQuality,_minDist);
	size_t features0Size = _features0.size();
	size_t features1Size = _features1.size();
	for(int i=0; i<_nSuperPixels; i++)
	{
		_features0.push_back(cv::Point2f(centers0[i].xy.x,centers0[i].xy.y));
		_features1.push_back(cv::Point2f(centers0[i].xy.x,centers1[i].xy.y));
	}
	// 2. track features
	cv::calcOpticalFlowPyrLK(gray, preGray, // 2 consecutive images
		_features0, // input point position in first image
		_matched0, // output point postion in the second image
		_status,    // tracking success
		_err);      // tracking error

	// 2. loop over the tracked points to reject the undesirables
	int k=0;

	for( int i= 0; i < _features0.size(); i++ ) {

		// do we keep this point?
		if (_status[i] == 1) {

			//m_features.data[(int)m_points[0][i].x+(int)m_points[0][i].y*m_oImgSize.width] = 0xff;
			// keep this point in vector
			_features0[k] = _features0[i];
			_matched0[k++] = _matched0[i];
		}
	}
	_features0.resize(k);
	_matched0.resize(k);

	cv::calcOpticalFlowPyrLK(preGray, gray, // 2 consecutive images
		_features1, // input point position in first image
		_matched1, // output point postion in the second image
		_status,    // tracking success
		_err);      // tracking error

	// 2. loop over the tracked points to reject the undesirables
	k=0;

	for( int i= 0; i < _features1.size(); i++ ) {

		// do we keep this point?
		if (_status[i] == 1) {

			//m_features.data[(int)m_points[0][i].x+(int)m_points[0][i].y*m_oImgSize.width] = 0xff;
			// keep this point in vector
			_features1[k] = _features1[i];
			_matched1[k++] = _matched1[i];
		}
	}
	_features1.resize(k);
	_matched1.resize(k);
	
	//ransac
	std::vector<uchar> inliers(_features0.size(),0);
	transM = cv::findHomography(_features0,_matched0,inliers,CV_RANSAC,0.1);
	cv::Scalar color(255,0,0);
	mask.create(_height,_width,CV_8UC1);
	mask = cv::Scalar(0);
	for(int i=0; i<_features0.size(); i++)
	{
		if (inliers[i]==1)
		{
			cv::circle(curImg,_features0[i],5,color);
			int k = _features0[i].x;
			int j = _features0[i].y;		
			int label = _labels0[k+ j*_width];
			//以原来的中心点为中心，step +2　为半径进行更新
			int radius = _step;
			for (int x = k- radius; x<= k+radius; x++)
			{
				for(int y = j - radius; y<= j+radius; y++)
				{
					if  (x<0 || x>_width-1 || y<0 || y> _height-1)
						continue;
					int idx = x+y*_width;
					//std::cout<<idx<<std::endl;
					if (_labels0[idx] == label )
					{		
						mask.data[idx] = 0xff;						
					}					
				}
			}
		}
	}
	inliers.resize(_features1.size());
	transM = cv::findHomography(_features1,_matched1,inliers,CV_RANSAC,0.1);
	for(int i=0; i<_features1.size(); i++)
	{
		if (inliers[i]==1)
		{
			cv::circle(curImg,_matched1[i],5,color);
			int k = _matched1[i].x;
			int j = _matched1[i].y;		
			int label = _labels0[k+ j*_width];
			//以原来的中心点为中心，step +2　为半径进行更新
			int radius = _step;
			for (int x = k- radius; x<= k+radius; x++)
			{
				for(int y = j - radius; y<= j+radius; y++)
				{
					if  (x<0 || x>_width-1 || y<0 || y> _height-1)
						continue;
					int idx = x+y*_width;
					//std::cout<<idx<<std::endl;
					if (_labels0[idx] == label )
					{		
						mask.data[idx] = 0xff;						
					}					
				}
			}
		}
	}
	//postProcess(img0,mask);
	
	////find most match superpixel
	//std::vector<int> matchedCount0(_nSuperPixels,0);
	//std::vector<int> matchedCount1(_nSuperPixels,0);
	////每个超像素的特征点id
	//std::vector<std::vector<int>> featuresPerSP0(_nSuperPixels);
	//std::vector<std::vector<int>> featuresPerSP1(_nSuperPixels);
	//for(int i=0; i< _features0.size(); i++)
	//{
	//	int ix = (int)_features0[i].x;
	//	int iy = (int)_features0[i].y;
	//	int label = _labels0[iy*_width+ix];
	//	matchedCount0[label]++;
	//	featuresPerSP0[label].push_back(i);
	//}
	//std::vector<int>::iterator itr = std::max_element(matchedCount0.begin(),matchedCount0.end());
	//std::vector<int>& f = featuresPerSP0[itr-matchedCount0.begin()];
	//cv::Scalar color(255,0,0);
	//for(int i=0; i<f.size(); i++)
	//{
	//	cv::circle(curImg,_features0[f[i]],2,color);
	//}

	//for(int i=0; i< _features1.size(); i++)
	//{
	//	int ix = (int)_features1[i].x;
	//	int iy = (int)_features1[i].y;
	//	int label = _labels0[iy*_width+ix];
	//	matchedCount1[label]++;
	//	featuresPerSP1[label].push_back(i);
	//}
	//itr = std::max_element(matchedCount1.begin(),matchedCount1.end());
	//f = featuresPerSP1[itr-matchedCount1.begin()];
	//for(int i=0; i<f.size(); i++)
	//{
	//	cv::circle(prevImg,_features1[f[i]],2,color);
	//}
	cv::Mat dmat(mask.size(),CV_8U);
	dmat = cv::Scalar(0);
	//求每个超像素inliers的密度
	int winSize = _step*10;
	for(int i=0; i< _nSuperPixels; i++)
	{		
		float2 center = centers0[i].xy;
		int ox = center.x;
		int oy = center.y;
		int bgInliers = 0;
		int nPixels(0);
		for(int m = ox-winSize; m<=ox+winSize; m++)
		{
			for(int n=oy-winSize; n<=oy+winSize; n++)
			{
				if (m>=0 && m<_width && n>=0 && n<_height)
				{
					if (mask.data[m+n*_width] == 0xff)
					{
						bgInliers++;
					}
					nPixels++;
				}
			}
		}
		float density = bgInliers*1.0/nPixels;
		if (density > 0.5)
			dmat.data[oy*_width+ox] = 0xff;
	}
	cv::imwrite("dmat.jpg",dmat);
	cv::imwrite("mask.jpg",mask);
}