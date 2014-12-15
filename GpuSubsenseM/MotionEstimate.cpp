#include "MotionEstimate.h"
#include "GpuSuperpixel.h"
#include <algorithm>
#include "timer.h"
#include <set>
#include <opencv2\opencv.hpp>
#include "MeanShift.h"
#include <fstream>
void postProcessSegments(Mat& img)
{
	cv::bitwise_not(img,img);
	int niters = 3;

	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy,imgHierarchy;
	
	Mat temp;
	
	//Mat edge(img.size(),CV_8U);	
	////cv::Canny(img,edge,100,300);
	dilate(img, img, Mat(), Point(-1,-1), niters);//膨胀，3*3的element，迭代次数为niters
	erode(img, img, Mat(), Point(-1,-1), niters*2);//腐蚀
	dilate(img, img, Mat(), Point(-1,-1), niters);
	findContours( img, contours, imgHierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE );//找轮廓

	
	
	if( contours.size() == 0 )
		return;
	img = cv::Scalar(0);
	
	double minArea = 50*50;
	Scalar color( 255, 255, 255 );
	//img = cv::Scalar(0);
	for( int i = 0; i< contours.size(); i++ )
	{
		const vector<Point>& c = contours[i];
		double area = fabs(contourArea(Mat(c)));
		if( area > minArea )
		{
			drawContours( img, contours, i, color, -1, 8, hierarchy, 0, Point() );			
		}
		
	}
	
}
//检测所求出前景的运动是否与背景一致，去掉错误前景
void MaskHomographyTest(cv::Mat& mCurr, cv::Mat& curr, cv::Mat & prev, cv::Mat& homography, float* distance)
{

	float threshold = 0.5;
	std::vector<cv::Point2f> currPoints, trackedPoints;
	std::vector<uchar> status; // status of tracked features
	std::vector<float> err;    // error in tracking
	for(int i=0; i<mCurr.cols; i++)
	{
		for(int j=0; j<mCurr.rows; j++)
			if(mCurr.data[i + j*mCurr.cols] == 0xff)
				currPoints.push_back(cv::Point2f(i,j));
	}
	if (currPoints.size() <=0)
		return;
	// 2. track features
	cv::calcOpticalFlowPyrLK(curr, prev, // 2 consecutive images
		currPoints, // input point position in first image
		trackedPoints, // output point postion in the second image
		status,    // tracking success
		err);      // tracking error

	// 2. loop over the tracked points to reject the undesirables
	int k=0;
	for( int i= 0; i < currPoints.size(); i++ ) {
		// do we keep this point?
		if (status[i] == 1) {
			// keep this point in vector
			currPoints[k] = currPoints[i];
			trackedPoints[k++] = trackedPoints[i];
		}
	}
	// eliminate unsuccesful points
	currPoints.resize(k);
	trackedPoints.resize(k);

	for(int i=0; i<k; i++)
	{
		cv::Point2f pt = currPoints[i];
		double* data = (double*)homography.data;
		float x = data[0]*pt.x + data[1]*pt.y + data[2];
		float y = data[3]*pt.x + data[4]*pt.y + data[5];
		float w = data[6]*pt.x + data[7]*pt.y + data[8];
		x /= w;
		y /= w;
		float d = abs(trackedPoints[i].x-x) + abs(trackedPoints[i].y - y);
		const size_t idx_char = (int)currPoints[i].x+(int)currPoints[i].y*mCurr.cols;
		//distance[idx_char]= d;
		if (d < threshold)
		{

			mCurr.data[idx_char] = 0x0;			

		}			
	}
}
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
float EDistance(const float4& f1, const float4& f2)
{
	float dx = f1.x - f2.x;
	float dy = f1.y - f2.y;
	float dz = f1.z - f2.z;
	//float dw = f1.w - f2.w;
	return sqrt(dx*dx + dy*dy + dz*dz);
}
float avgDist(int widht, int height,int step,int nPixel,const SLICClusterCenter* centers)
{
	float avgE = 0;
	size_t count = 0;	
	int xStep = ((widht+ step-1) / step);
	for(int i=0; i<nPixel; i++)
	{
		if (centers[i].nPoints > 0)
		{
			if (i-1>=0)
			{
				if (centers[i-1].nPoints > 0 )
				{
					avgE += EDistance(centers[i].rgb,centers[i-1].rgb);
					count ++;
				}
				if(i+xStep-1<nPixel && centers[i+xStep-1].nPoints > 0)
				{
					avgE += EDistance(centers[i].rgb,centers[i+xStep-1].rgb);
					count ++;
				}
				if (i-xStep-1>=0 && centers[i-xStep-1].nPoints > 0)
				{
					avgE += EDistance(centers[i].rgb,centers[i-xStep-1].rgb);
					count ++;
				}

			}
			if(i+1<nPixel)
			{
				if (centers[i+1].nPoints > 0)
				{
					avgE += EDistance(centers[i].rgb,centers[i+1].rgb);
					count ++;

				}
				if(i+xStep+1<nPixel && centers[i+xStep+1].nPoints > 0)
				{
					avgE += EDistance(centers[i].rgb,centers[i+xStep+1].rgb);
					count ++;
				}
				if (i-xStep+1>=0 && centers[i-xStep+1].nPoints >0)
				{
					avgE += EDistance(centers[i].rgb,centers[i-xStep+1].rgb);
					count ++;
				}				
			}

			if (i-xStep>=0 && centers[i-xStep].nPoints > 0)
			{
				avgE += EDistance(centers[i].rgb,centers[i-xStep].rgb);
				count ++;
			}			

			if(i+xStep<nPixel && centers[i+xStep].nPoints > 0)
			{
				avgE += EDistance(centers[i].rgb,centers[i+xStep].rgb);
				count ++;
			}	

		}	
	}

	avgE /= count;
	return avgE;
}
float OstuThreshold(int width, int height, int step, const SLICClusterCenter* centers)
{
	size_t histogram[256] = {0};
	int spSize = (width+step-1)/step * (height+step-1)/step;
	for(int i=0; i<spSize; i++)
	{
		float4 rgb = centers[i].rgb;
		uchar gray = (uchar)((rgb.x+rgb.y+rgb.z)/3);
		histogram[gray]++;
	}
	size_t sum = 0;
	for(int i=0; i<256; i++)
	{
		sum += i*histogram[i];
	}
	size_t sumB(0),wB(0),wF(0);
	size_t mB,mF;

	float max(0.f),between(0.f), threshold1(0.f),threshold2(0.f);
	for(int i=0; i<256; i++)
	{
		wB += histogram[i];
		if (wB == 0)
			continue;
		wF = spSize - wB;
		if (wF ==0 )
			break;
		sumB += i*histogram[i];
		mB = sumB/wB;
		mF = (sum - sumB)/wF;
		between = wB * wF * pow(mB-mF,2.0);
		if(between >= max)
		{
			threshold1 = i;
			if (between > max)
				threshold2 = i;
			max = between;
		}
	}
	return ( threshold1 + threshold2 ) / 2.0;
}
void SuperPixelRegionGrowing(int width, int height, int step,std::vector<int>& spLabels, const int*  labels, const SLICClusterCenter* centers, cv::Mat& result,int threshold)
{
	if (result.empty())
	{
		result.create(height,width,CV_8U);
		
	}
	result = cv::Scalar(0);
	const int dx4[] = {-1,0,1,0};
	const int dy4[] = {0,-1,0,1};
	//const int dx8[8] = {-1, -1,  0,  1, 1, 1, 0, -1};
	//const int dy8[8] = { 0, -1, -1, -1, 0, 1, 1,  1};
	int spWidth = (width+step-1)/step;
	int spHeight = (height+step-1)/step;
	float pixDist(0);
	float regMaxDist = threshold;
	int regSize(0);
	int imgSize = spWidth*spHeight;
	char* visited = new char[imgSize];
	char* segmented = new char[imgSize];
	memset(segmented,0,imgSize);
	std::vector<cv::Point2i> neighbors;
	float4 regMean;
	std::set<int> resLabels;
	//nih::Timer timer;
	//timer.start();
	for(int i=0; i<spLabels.size(); i++)
	{
		memset(visited ,0,imgSize);
		resLabels.insert(spLabels[i]);
		SLICClusterCenter cc = centers[spLabels[i]];
		int k = cc.xy.x;
		int j = cc.xy.y;		
		int label = labels[k+ j*width];
		int ix = label%spWidth;
		int iy = label/spWidth;
		pixDist = 0;
		regSize = 1;
		
		neighbors.clear();
	
		regMean = cc.rgb;
		while(pixDist < regMaxDist && regSize<imgSize)
		{
			for(int d=0; d<4; d++)
			{
				int x = ix+dx4[d];
				int y = iy + dy4[d];
				if (x>=0 && x<spWidth && y>=0 && y<spHeight && !visited[x+y*spWidth] && !segmented[x+y*spWidth])
				{
					neighbors.push_back(cv::Point2i(x,y));
					visited[x+y*spWidth] = true;
				}
			}
			int idxMin = 0;
			pixDist = 255;
			if (neighbors.size() == 0)
				break;
			for(int j=0; j<neighbors.size(); j++)
			{
				size_t idx = neighbors[j].x+neighbors[j].y*spWidth;
				float4 rgb = centers[idx].rgb;
				float dx = rgb.x - regMean.x;
				float dy = rgb.y -regMean.y;
				float dz = rgb.z - regMean.z;
				//float dist = (abs(dx) + abs(dy)+ abs(dz))/3;
				float dist = sqrt(dx*dx + dy*dy + dz*dz);
				if (dist < pixDist)
				{
					pixDist = dist;
					idxMin = j;
				}				
			}
			
			ix = neighbors[idxMin].x;
			iy = neighbors[idxMin].y;
			int minIdx =ix + iy*spWidth;
			float4 rgb = centers[minIdx].rgb;
			//std::cout<<ix<<" "<<iy<<" added ,regMean = "<< regMean<<" pixDist "<<pixDist<<std::endl;
			regMean.x = (rgb.x + regMean.x*regSize )/(regSize+1);
			regMean.y = (rgb.y + regMean.y*regSize )/(regSize+1);
			regMean.z = (rgb.z + regMean.z*regSize )/(regSize+1);
			regSize++;
			int label = minIdx;
			resLabels.insert(label);
			segmented[minIdx] = 1;
			//result.data[minIdx] = 0xff;
			//smask.data[minIdx] = 0xff;
			neighbors[idxMin] = neighbors[neighbors.size()-1];
			neighbors.pop_back();
		}
	}
	/*timer.stop();
	std::cout<<"region growing "<<timer.seconds()<<std::endl;
	timer.start();*/
	for(int i=0; i<width; i++)
	{
		for(int j=0; j<height; j++)
		{
			int idx = i+j*width;
			if (resLabels.find(labels[idx]) != resLabels.end())
				result.data[idx] = 0xff;
		}
	}
	/*timer.stop();
	std::cout<<"result updating "<<timer.seconds()<<std::endl;*/
	delete[] visited;
	delete[] segmented;
}
void SPRGPostProcess(int width, int height, int step, int spWidth, int spHeight,int flagLabel, int nlabel, std::vector<int>& newLabels)
{
	const int dx4[] = {-1,0,1,0};
	const int dy4[] = {0,-1,0,1};
	int stack[1024];
	int p(0);
	int* visited = new int[spWidth*spHeight];
	memset(visited,0,spWidth*spHeight*sizeof(int));
	for(int i=0; i<spWidth; i++)
	{
		for(int j=0; j<spHeight; j++)
		{
			int idx = i+j*spWidth;
			if (newLabels[idx] == flagLabel)
			{
				for(int d = 0; d<4; d++)
				{
					int ix = dx4[d] + i;
					int iy = dy4[d] +j;
					if(ix>=0 && ix < spWidth && iy >=0 && iy <spHeight)
					{
						if (newLabels[ix+iy*spWidth] != flagLabel)
						{
							newLabels[idx] = newLabels[ix+iy*spWidth];

							break;
						}
					}
				}
				//p=0;
				//stack[p++] = idx;
				//while(p>=0)
				//{
				//	int id = stack[--p];
				//	newLabels[id] = nlabel;
				//	visited[id] = 1;
				//	for(int d = 0; d<4; d++)
				//	{

				//		int ix = dx4[d] + i;
				//		int iy = dy4[d] +j;
				//		if(ix>=0 && ix < spWidth && iy >=0 && iy <spHeight)
				//		{
				//			int nid = ix+iy*spWidth;
				//			if (!visited[nid] && newLabels[nid] == flagLabel)
				//				stack[p++] = idx;
				//		}


				//	}
				//}

			}
			nlabel++;
		}
	}
}
void SuperPixelRGSegment(int width, int height, int step,const int*  labels, const SLICClusterCenter* centers,int threshold, int*& segmented)
{
	const int dx4[] = {-1,0,1,0};
	const int dy4[] = {0,-1,0,1};
	//const int dx8[8] = {-1, -1,  0,  1, 1, 1, 0, -1};
	//const int dy8[8] = { 0, -1, -1, -1, 0, 1, 1,  1};
	int spWidth = (width+step-1)/step;
	int spHeight = (height+step-1)/step;
	float pixDist(0);
	float regMaxDist = threshold;
	int regSize(0);
	//当前新标签
	int curLabel(1);
	int imgSize = spWidth*spHeight;
	char* visited = new char[imgSize];
	memset(visited ,0,imgSize);
	memset(segmented,0,sizeof(int)*width*height);
	std::vector<cv::Point2i> neighbors;
	float4 regMean;
	//region growing 后的新label
	std::vector<int> newLabels;
	newLabels.resize(imgSize);
	//nih::Timer timer;
	//timer.start();
	std::set<int> boundarySet;
	boundarySet.insert(rand()%imgSize);
	
	std::vector<int> labelGroup;
	
	while(!boundarySet.empty())
	{
		labelGroup.clear();
		std::set<int>::iterator itr = boundarySet.begin();
		int label = *itr;
		labelGroup.push_back(label);
		//newLabels[label] = curLabel;
		boundarySet.erase(itr);
		SLICClusterCenter cc = centers[label];
		int k = cc.xy.x;
		int j = cc.xy.y;		
		int ix = label%spWidth;
		int iy = label/spWidth;
		pixDist = 0;
		regSize = 1;
		//segmented[ix+iy*spWidth] = curLabel;
		neighbors.clear();
		regMean = cc.rgb;
		
		
		while(pixDist < regMaxDist && regSize<imgSize)
		{
			for(int d=0; d<4; d++)
			{
				int x = ix+dx4[d];
				int y = iy + dy4[d];
				if (x>=0 && x<spWidth && y>=0 && y<spHeight && !visited[x+y*spWidth])
				{
					neighbors.push_back(cv::Point2i(x,y));
					visited[x+y*spWidth] = true;
				}
			}
			int idxMin = 0;
			pixDist = 255;
			if (neighbors.size() == 0)
				break;

			for(int j=0; j<neighbors.size(); j++)
			{
				size_t idx = neighbors[j].x+neighbors[j].y*spWidth;
				float4 rgb = centers[idx].rgb;
				float dx = rgb.x - regMean.x;
				float dy = rgb.y -regMean.y;
				float dz = rgb.z - regMean.z;
				//float dist = (abs(dx) + abs(dy)+ abs(dz))/3;
				float dist = sqrt(dx*dx + dy*dy + dz*dz);
				if (dist < pixDist)
				{
					pixDist = dist;
					idxMin = j;
				}				
			}

			ix = neighbors[idxMin].x;
			iy = neighbors[idxMin].y;
			int minIdx =ix + iy*spWidth;			
			float4 rgb = centers[minIdx].rgb;
			//std::cout<<ix<<" "<<iy<<" added ,regMean = "<< regMean<<" pixDist "<<pixDist<<std::endl;
			regMean.x = (rgb.x + regMean.x*regSize )/(regSize+1);
			regMean.y = (rgb.y + regMean.y*regSize )/(regSize+1);
			regMean.z = (rgb.z + regMean.z*regSize )/(regSize+1);
			regSize++;
			labelGroup.push_back(minIdx);
			
			visited[minIdx] = true;
			/*segmented[minIdx] = k;*/
			//result.data[minIdx] = 0xff;
			//smask.data[minIdx] = 0xff;
			neighbors[idxMin] = neighbors[neighbors.size()-1];
			neighbors.pop_back();
		}
		//孤立点
		if (regSize <2)
		{
			for(int i=0; i<labelGroup.size(); i++)
			{
				newLabels[labelGroup[i]] = 0;
				//std::cout<<labels[labelGroup[i]]<<std::endl;
			}
		}
		else
		{
			for(int i=0; i<labelGroup.size(); i++)
			{
				newLabels[labelGroup[i]] = curLabel;
			}
			curLabel++;
		}
		for(int i=0; i<neighbors.size(); i++)
		{
			int label = neighbors[i].x + neighbors[i].y*spWidth;
			if (boundarySet.find(label) == boundarySet.end())
				boundarySet.insert(label);
			
		}
	}
	SPRGPostProcess(width,height,step,spWidth,spHeight,0,curLabel,newLabels);
	for(int i=0; i<newLabels.size(); i++)
	{
		int x = centers[i].xy.x;
		int y = centers[i].xy.y;
		for(int dx= -step; dx<=step; dx++)
		{
			for(int dy = -step; dy<=step; dy++)
			{
				int sx = x+dx;
				int sy = y + dy;
				if(sx>=0 && sx<width && sy>=0 && sy<height)
				{
					int idx = sx+sy*width;
					if (labels[idx] == i)
						segmented[idx] = newLabels[i];
				}
			}
		}

	}

	
	delete[] visited;
	//delete[] segmented;

}
void RegionGrowing(std::vector<cv::Point2f>& seeds,const cv::Mat& img, cv::Mat& result)
{	
	const int nx[] = {-1,0,1,0};
	const int ny[] = {0,-1,0,1};
	cv::Mat gray;
	cv::cvtColor(img,gray,CV_BGR2GRAY);
	int width = img.cols;
	int height = img.rows;
	float pixDist(0);
	float regMaxDist = 0.2*255;
	int regSize(0);
	int imgSize = width*height;
	char* visited = new char[imgSize];
	memset(visited ,0,imgSize);
	std::vector<cv::Point2i> neighbors;
	float3 regMean;
	char filename[50];
	for(int i=0; i<seeds.size(); i++)
	{
		/*cv::Mat smask(img.size(),CV_8U);
		smask = cv::Scalar(0);*/
		pixDist = 0;
		regSize = 1;
		
		neighbors.clear();
		int ix = seeds[i].x;
		int iy = seeds[i].y;

		sprintf(filename,"%d_%d.jpg",ix,iy);
		size_t idx_uchar = (ix+iy*width);
		size_t idx_rgb = idx_uchar*3;
		regMean = make_float3(img.data[idx_rgb],img.data[idx_rgb+1],img.data[idx_rgb+2]);
		while(pixDist < regMaxDist && regSize<imgSize)
		{
			for(int d=0; d<4; d++)
			{
				int x = ix+nx[d];
				int y = iy + ny[d];
				if (!visited[x+y*width] && x>=0 && x<width && y>=0 && y<height)
				{
					neighbors.push_back(cv::Point2i(x,y));
					visited[x+y*width] = true;
				}
			}
			int idxMin = 0;
			pixDist = 255;
			for(int j=0; j<neighbors.size(); j++)
			{
				size_t idx = neighbors[j].x+neighbors[j].y*width;
				size_t idx_rgb= idx*3;
				float dist = (abs(img.data[idx_rgb] - regMean.x) + abs(img.data[idx_rgb+1]-regMean.y )+ abs(img.data[idx_rgb+2]-regMean.z))/3;
				if (dist < pixDist)
				{
					pixDist = dist;
					idxMin = j;
				}				
			}
			if (neighbors.size() == 0)
				break;
			ix = neighbors[idxMin].x;
			iy = neighbors[idxMin].y;
			int minIdx =ix + iy*width;
			size_t minIdx_rgb = minIdx*3;
			//std::cout<<ix<<" "<<iy<<" added ,regMean = "<< regMean<<" pixDist "<<pixDist<<std::endl;
			regMean.x = (img.data[minIdx_rgb] + regMean.x*regSize )/(regSize+1);
			regMean.y = (img.data[minIdx_rgb+1] + regMean.y*regSize )/(regSize+1);
			regMean.z = (img.data[minIdx_rgb+2] + regMean.z*regSize )/(regSize+1);
			regSize++;
			result.data[minIdx] = 0xff;
			//smask.data[minIdx] = 0xff;
			neighbors[idxMin] = neighbors[neighbors.size()-1];
			neighbors.pop_back();
		}
		/*cv::bitwise_and(smask,gray,smask);
		cv::imwrite(filename,smask);*/
	}
}
void MotionEstimate::KLT(Mat& curImg, Mat& prevImg)
{
	cv::Mat gray,preGray;
	cv::cvtColor(curImg,gray,CV_BGR2GRAY);
	cv::cvtColor(prevImg,preGray,CV_BGR2GRAY);
	cv::goodFeaturesToTrack(gray,_features0,_maxCorners,_dataQuality,_minDist);
	/*cv::goodFeaturesToTrack(preGray,_features1,_maxCorners,_dataQuality,_minDist);*/
	size_t features0Size = _features0.size();
	//size_t features1Size = _features1.size();
	// 2. track features
	cv::calcOpticalFlowPyrLK(gray, preGray, // 2 consecutive images
		_features0, // input point position in first image
		_matched0, // output point postion in the second image
		_status,    // tracking success
		_err);      // tracking error

	// 2. loop over the tracked points to reject the undesirables
	int k=0;

	for( int i= 0; i < features0Size; i++ ) {

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
}
void MotionEstimate::EstimateMotionMeanShift(Mat& curImg, Mat& prevImg, Mat& transM, Mat& mask)
{
	KLT(curImg,prevImg);
	MeanShift(curImg,_labels0);
	std::vector<uchar> inliers(_features0.size(),0);
	transM = cv::findHomography(_features0,_matched0,inliers,CV_RANSAC,0.08);
	cv::Scalar color(255,0,0);
	mask.create(_height,_width,CV_8UC1);
	mask = cv::Scalar(0);
	std::vector<cv::Point2f> bgPoints;
	std::set<int> spLabels;
	int spWidth = (_width+_step-1)/_step;
	for(int i=0; i<_features0.size(); i++)
	{
		if (inliers[i]==1)
		{
			int k = _features0[i].x;
			int j = _features0[i].y;		
			int label = _labels0[k+ j*_width];
			spLabels.insert(label);
		}
	}
	for(int i=0; i<_height; i++)
	{
		for(int j=0; j<_width; j++)
		{
			int idx = i*_width+j;
			if (spLabels.find(_labels0[idx])!= spLabels.end())
			{
				mask.data[idx] = 0xff;
			
			}
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

			/*_imgData1[i + j*_width].x = img1.data[idx];
			_imgData1[i + j*_width].y = img1.data[idx+ img1.elemSize1()];
			_imgData1[i + j*_width].z = img1.data[idx+2*img1.elemSize1()];
			_imgData1[i + j*_width].w = img1.data[idx+3*img1.elemSize1()];*/
		}
	}
	
	/*centers1 = new SLICClusterCenter[_nSuperPixels];*/
	_gs->SuperpixelLattice(_imgData0,num,_labels0,_centers0);
	//_gs->Superpixel(_imgData1,num,_labels1,centers1);
	
	//Good Features
	cv::Mat gray,preGray;
	cv::cvtColor(curImg,gray,CV_BGR2GRAY);
	cv::cvtColor(prevImg,preGray,CV_BGR2GRAY);
	cv::goodFeaturesToTrack(gray,_features0,_maxCorners,_dataQuality,_minDist);
	/*cv::goodFeaturesToTrack(preGray,_features1,_maxCorners,_dataQuality,_minDist);*/
	size_t features0Size = _features0.size();
	//size_t features1Size = _features1.size();
	for(int i=0; i<_nSuperPixels; i++)
	{
		_features0.push_back(cv::Point2f(_centers0[i].xy.x,_centers0[i].xy.y));
		//_features1.push_back(cv::Point2f(centers0[i].xy.x,centers1[i].xy.y));
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

	//cv::calcOpticalFlowPyrLK(preGray, gray, // 2 consecutive images
	//	_features1, // input point position in first image
	//	_matched1, // output point postion in the second image
	//	_status,    // tracking success
	//	_err);      // tracking error

	//// 2. loop over the tracked points to reject the undesirables
	//k=0;

	//for( int i= 0; i < _features1.size(); i++ ) {

	//	// do we keep this point?
	//	if (_status[i] == 1) {

	//		//m_features.data[(int)m_points[0][i].x+(int)m_points[0][i].y*m_oImgSize.width] = 0xff;
	//		// keep this point in vector
	//		_features1[k] = _features1[i];
	//		_matched1[k++] = _matched1[i];
	//	}
	//}
	//_features1.resize(k);
	//_matched1.resize(k);
	
	//ransac
	std::vector<uchar> inliers(_features0.size(),0);
	transM = cv::findHomography(_features0,_matched0,inliers,CV_RANSAC,0.08);
	cv::Scalar color(255,0,0);
	mask.create(_height,_width,CV_8UC1);
	mask = cv::Scalar(0);
	std::vector<cv::Point2f> bgPoints;
	std::vector<int> spLabels;
	int spWidth = (_width+_step-1)/_step;
	for(int i=0; i<_features0.size(); i++)
	{
		if (inliers[i]==1)
		{
			/*bgPoints.push_back(_features0[i]);*/
			cv::circle(curImg,_features0[i],5,color);
			int k = _features0[i].x;
			int j = _features0[i].y;		
			int label = _labels0[k+ j*_width];
			spLabels.push_back(label);
			////添加周围邻域superpixel
			//if (label+1<_nSuperPixels)
			//	spLabels.push_back(label+1);
			//if (label-1>=0)
			//	spLabels.push_back(label-1);
			//if (label+spWidth <_nSuperPixels)
			//	spLabels.push_back(label+spWidth);
			//if(label-spWidth >=0)
			//	spLabels.push_back(label-spWidth);
			//以原来的中心点为中心，step +2　为半径进行更新
			//int radius = _step;
			//for (int x = k- radius; x<= k+radius; x++)
			//{
			//	for(int y = j - radius; y<= j+radius; y++)
			//	{
			//		if  (x<0 || x>_width-1 || y<0 || y> _height-1)
			//			continue;
			//		
			//		int idx = x+y*_width;
			//		//std::cout<<idx<<std::endl;
			//		if (_labels0[idx] == label )
			//		{		
			//			bgPoints.push_back(cv::Point2f(x,y));
			//			//mask.data[idx] = 0xff;						
			//		}					
			//	}
			//}
		}
		else
		{
			//cv::circle(curImg,_features0[i],5,cv::Scalar(0,0,255));
		}
	}
	mask.create(curImg.size(),CV_8U);
	mask = cv::Scalar(0);
	//RegionGrowing(bgPoints,curImg,mask);
	/*float threshold = OstuThreshold(_width,_height,_step,centers0);*/
	

	//inliers.resize(_features1.size());
	//cv::findHomography(_features1,_matched1,inliers,CV_RANSAC,0.1);
	//for(int i=0; i<_features1.size(); i++)
	//{
	//	if (inliers[i]==1)
	//	{
	//		cv::circle(curImg,_matched1[i],5,color);
	//		/*int k = _matched1[i].x;
	//		int j = _matched1[i].y;		
	//		int label = _labels0[k+ j*_width];
	//		spLabels.push_back(label);*/
	//		////以原来的中心点为中心，step +2　为半径进行更新
	//		//int radius = _step;
	//		//for (int x = k- radius; x<= k+radius; x++)
	//		//{
	//		//	for(int y = j - radius; y<= j+radius; y++)
	//		//	{
	//		//		if  (x<0 || x>_width-1 || y<0 || y> _height-1)
	//		//			continue;
	//		//		int idx = x+y*_width;
	//		//		//std::cout<<idx<<std::endl;
	//		//		if (_labels0[idx] == label )
	//		//		{		
	//		//			mask.data[idx] = 0xff;						
	//		//		}					
	//		//	}
	//		//}
	//	}
	//}
	float threshold = avgDist(_width,_height,_step,_nSuperPixels,_centers0);
	std::cout<<"threshold= "<<threshold<<std::endl;
	SuperPixelRegionGrowing(_width,_height,_step,spLabels,_labels0,_centers0,mask,threshold*0.8);
	
	postProcessSegments(mask);
	//MaskHomographyTest(mask,gray,preGray,transM,NULL);
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
	//cv::Mat dmat(mask.size(),CV_8U);
	//dmat = cv::Scalar(0);
	////求每个超像素inliers的密度
	//int winSize = _step*10;
	//for(int i=0; i< _nSuperPixels; i++)
	//{		
	//	float2 center = centers0[i].xy;
	//	int ox = center.x;
	//	int oy = center.y;
	//	int bgInliers = 0;
	//	int nPixels(0);
	//	for(int m = ox-winSize; m<=ox+winSize; m++)
	//	{
	//		for(int n=oy-winSize; n<=oy+winSize; n++)
	//		{
	//			if (m>=0 && m<_width && n>=0 && n<_height)
	//			{
	//				if (mask.data[m+n*_width] == 0xff)
	//				{
	//					bgInliers++;
	//				}
	//				nPixels++;
	//			}
	//		}
	//	}
	//	float density = bgInliers*1.0/nPixels;
	//	if (density > 0.5)
	//		dmat.data[oy*_width+ox] = 0xff;
	//}
	//cv::imwrite("dmat.jpg",dmat);
	//cv::imwrite("mask.jpg",mask);
	
}
void OpticalFlowHistogram(cv::Mat& flow,
	std::vector<float>& histogram, std::vector<std::vector<int>>& ids, int DistSize = 16,int thetaSize = 16)
{
	//直方图共256个bin，其中根据光流强度分16个bin，每个bin根据光流方向分16个bin
	int binSize = DistSize * thetaSize;
	histogram.resize(binSize);
	ids.resize(binSize);
	for(int i=0; i<binSize; i++)
		ids[i].clear();
	memset(&histogram[0],0,sizeof(float)*binSize);
	

	cv::Mat xy[2];
	cv::split(flow, xy);

	//calculate angle and magnitude
	cv::Mat magnitude, angle;
	cv::cartToPolar(xy[0], xy[1], magnitude, angle, true);

	//translate magnitude to range [0;1]
	double mag_max;
	cv::minMaxLoc(magnitude, 0, &mag_max);
	magnitude.convertTo(magnitude, -1, 1.0/mag_max);

	
	float stepR = 1.0/DistSize;
	float stepT = 360.0/thetaSize;
	double* magPtr = (double*)magnitude.data;
	double* angPtr = (double*)angle.data;
	for(int i = 0; i<magnitude.rows; i++)
	{
		for(int j=0; j<magnitude.cols; j++)
		{
			int idx = i*magnitude.cols+j;
			int r = (int)(magPtr[idx]/stepR);
			int t = (int)(angPtr[idx]/stepT);
			r = r>DistSize? DistSize:r;
			t = t>thetaSize? thetaSize:t;
			idx = t*DistSize+r;
			//std::cout<<idx<<std::endl;
			histogram[idx]++;
			ids[idx].push_back(idx);

		}
	}
	
}
void MotionEstimate::EstimateMotionHistogram( Mat& curImg,  Mat& prevImg, Mat& transM, Mat& mask)
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

			/*_imgData1[i + j*_width].x = img1.data[idx];
			_imgData1[i + j*_width].y = img1.data[idx+ img1.elemSize1()];
			_imgData1[i + j*_width].z = img1.data[idx+2*img1.elemSize1()];
			_imgData1[i + j*_width].w = img1.data[idx+3*img1.elemSize1()];*/
		}
	}
	
	/*centers1 = new SLICClusterCenter[_nSuperPixels];*/
	_gs->SuperpixelLattice(_imgData0,num,_labels0,_centers0);
	//_gs->Superpixel(_imgData1,num,_labels1,centers1);
	
	cv::Mat flow;
	cv::calcOpticalFlowSF(curImg,prevImg,flow,3,2,4);
	
	cv::Scalar color(255,0,0);
	mask.create(_height,_width,CV_8UC1);
	mask = cv::Scalar(0);
	std::vector<cv::Point2f> bgPoints;
	std::vector<int> spLabels;
	int spWidth = (_width+_step-1)/_step;
	
	mask.create(curImg.size(),CV_8U);
	mask = cv::Scalar(0);
	//RegionGrowing(bgPoints,curImg,mask);
	/*float threshold = OstuThreshold(_width,_height,_step,centers0);*/
	

	//inliers.resize(_features1.size());
	//cv::findHomography(_features1,_matched1,inliers,CV_RANSAC,0.1);
	//for(int i=0; i<_features1.size(); i++)
	//{
	//	if (inliers[i]==1)
	//	{
	//		cv::circle(curImg,_matched1[i],5,color);
	//		/*int k = _matched1[i].x;
	//		int j = _matched1[i].y;		
	//		int label = _labels0[k+ j*_width];
	//		spLabels.push_back(label);*/
	//		////以原来的中心点为中心，step +2　为半径进行更新
	//		//int radius = _step;
	//		//for (int x = k- radius; x<= k+radius; x++)
	//		//{
	//		//	for(int y = j - radius; y<= j+radius; y++)
	//		//	{
	//		//		if  (x<0 || x>_width-1 || y<0 || y> _height-1)
	//		//			continue;
	//		//		int idx = x+y*_width;
	//		//		//std::cout<<idx<<std::endl;
	//		//		if (_labels0[idx] == label )
	//		//		{		
	//		//			mask.data[idx] = 0xff;						
	//		//		}					
	//		//	}
	//		//}
	//	}
	//}
	float threshold = avgDist(_width,_height,_step,_nSuperPixels,_centers0);
	std::cout<<"threshold= "<<threshold<<std::endl;
	SuperPixelRegionGrowing(_width,_height,_step,spLabels,_labels0,_centers0,mask,threshold*0.8);
	
	postProcessSegments(mask);
	//MaskHomographyTest(mask,gray,preGray,transM,NULL);
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
	//cv::Mat dmat(mask.size(),CV_8U);
	//dmat = cv::Scalar(0);
	////求每个超像素inliers的密度
	//int winSize = _step*10;
	//for(int i=0; i< _nSuperPixels; i++)
	//{		
	//	float2 center = centers0[i].xy;
	//	int ox = center.x;
	//	int oy = center.y;
	//	int bgInliers = 0;
	//	int nPixels(0);
	//	for(int m = ox-winSize; m<=ox+winSize; m++)
	//	{
	//		for(int n=oy-winSize; n<=oy+winSize; n++)
	//		{
	//			if (m>=0 && m<_width && n>=0 && n<_height)
	//			{
	//				if (mask.data[m+n*_width] == 0xff)
	//				{
	//					bgInliers++;
	//				}
	//				nPixels++;
	//			}
	//		}
	//	}
	//	float density = bgInliers*1.0/nPixels;
	//	if (density > 0.5)
	//		dmat.data[oy*_width+ox] = 0xff;
	//}
	//cv::imwrite("dmat.jpg",dmat);
	//cv::imwrite("mask.jpg",mask);
	
}
void TestRegioinGrowingSegment()
{
	char fileName[100];
	int start = 1;
	int end = 1130;
	cv::Mat curImg,mask;
	int _width = 640;
	int _height = 480;
	int _step = 5;
	GpuSuperpixel gs(_width,_height,_step);
	int spWidth = (_width+_step-1)/_step;
	int spHeight = (_height+_step-1)/_step;
	int spSize = spWidth*spHeight;
	int* labels = new int[_width*_height];
	SLICClusterCenter* centers= new SLICClusterCenter[spSize];
	uchar4* _imgData0 = new uchar4[_width*_height];
	int* segment = new int[_width*_height];
	mask.create(_height,_width,CV_8UC3);
	std::vector<int> color(spSize);
	CvRNG rng= cvRNG(cvGetTickCount());
	color[0] = 0;
	for(int i=1;i<spSize;i++)
		color[i] = cvRandInt(&rng);
	for(int i=start; i<=end; i++)
	{
		sprintf(fileName,"..//moseg//people1//in%06d.jpg",i);
		curImg = cv::imread(fileName);
		cv::blur(curImg,curImg,cv::Size(3,3));
		cv::cvtColor(curImg,curImg,CV_BGR2BGRA);
		for(int i=0; i< _width; i++)
		{		
			for(int j=0; j<_height; j++)
			{
				int idx = curImg.step[0]*j + curImg.step[1]*i;
				int id = i + j*_width;
				_imgData0[id].x = curImg.data[idx];
				_imgData0[id].y = curImg.data[idx+ curImg.elemSize1()];
				_imgData0[id].z = curImg.data[idx+2*curImg.elemSize1()];
				_imgData0[id].w = curImg.data[idx+3*curImg.elemSize1()];						
			}
		}
		int num(0);		
		gs.SuperpixelLattice(_imgData0,num,labels,centers);
		/*std::ofstream file("label.txt");
		for(int j=0; j<_height; j++)
		{
			for(int i=0; i<_width; i++)
			{
				file<<labels[i+j*_height]<<"\t";
			}
			file<<"\n";
		}
		file.close();*/
		SuperPixelRGSegment(_width,_height,_step,labels,centers,28,segment);
		// Draw random color
		for(int i=0;i<_height;i++)
			for(int j=0;j<_width;j++)
			{ 
				int cl = segment[i*_width+j];
				((uchar *)(mask.data + i*mask.step.p[0]))[j*mask.step.p[1] + 0] = (color[cl])&255;
				((uchar *)(mask.data + i*mask.step.p[0]))[j*mask.step.p[1] + 1] = (color[cl]>>8)&255;
				((uchar *)(mask.data + i*mask.step.p[0]))[j*mask.step.p[1] + 2] = (color[cl]>>16)&255;
			}
			cv::Mat fmask;
			/*cv::bilateralFilter(mask,fmask,5,10,2.5);*/
			sprintf(fileName,".//segment//people1//features%06d.jpg",i);
			cv::imwrite(fileName,mask);

	}
	delete[] labels;
	delete[] centers;
	delete[] _imgData0;
	delete[] segment;
}