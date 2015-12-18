#include "FeaturePointRefine.h"
#include "findHomography.h"
#include "Common.h"

void FeaturePointsRefineRANSAC(std::vector<cv::Point2f>& vf1, std::vector<cv::Point2f>& vf2,cv::Mat& homography,float threshold)
{
	std::vector<uchar> inliers(vf1.size());
	homography = cv::findHomography(
		cv::Mat(vf1), // corresponding
		cv::Mat(vf2), // points
		inliers, // outputted inliers matches
		CV_RANSAC, // RANSAC method
		threshold); // max distance to reprojection point
	int k=0;
	for(int i=0; i<vf1.size(); i++)
	{
		if (inliers[i] ==1)
		{
			vf1[k] = vf1[i];
			vf2[k] = vf2[i];
			k++;
		}
	}
	vf1.resize(k);
	vf2.resize(k);
}
//nf是特征点数量，vf1中前面nf个是特征点，后面是超像素中心
void FeaturePointsRefineRANSAC(int& nf, std::vector<cv::Point2f>& vf1, std::vector<cv::Point2f>& vf2,cv::Mat& homography,float threshold)
{
	std::vector<uchar> inliers(vf1.size());

	homography = cv::findHomography(
		cv::Mat(vf1), // corresponding
		cv::Mat(vf2), // points
		inliers, // outputted inliers matches
		CV_RANSAC, // RANSAC method
		threshold); // max distance to reprojection point
	int k=0;
	for(int i=0; i<nf; i++)
	{
		if (inliers[i] ==1)
		{
			vf1[k] = vf1[i];
			vf2[k] = vf2[i];
			k++;
		}
	}
	int tmp = k;
	for(int i=nf; i<vf1.size(); i++)
	{
		if (inliers[i] ==1)
		{
			vf1[k] = vf1[i];
			vf2[k] = vf2[i];
			k++;
		}
	}
	nf = tmp;
	vf1.resize(k);
	vf2.resize(k);
}
void RelFlowRefine(std::vector<cv::Point2f>& features1, std::vector<cv::Point2f>&features0, float threshold)
{
	std::vector<uchar> inliers;
	int id;
	RelFlowRefine(features1, features0, inliers, id, threshold);

	int k(0);
	for (size_t i = 0; i < inliers.size(); i++)
	{
		if (inliers[i] == 1)
		{ 
			features0[k] = features0[i];
			features1[k] = features1[i];
			k++;
		}
			
	}
	features0.resize(k);
	features1.resize(k);
}
void IterativeOpticalFlowHistogram(std::vector<cv::Point2f>& f1, std::vector<cv::Point2f>& f2,
	std::vector<float>& histogram, std::vector<std::vector<int>>& ids, float ratioMax, float ratioMin)
{
	//直方图共DistSize * thetaSize个bin，其中根据光流强度分DistSize个bin，每个bin根据光流方向分thetaSize个bin
	int binSize; 
	float thetaSize = 60;
	float DistSize = 60;
	std::vector<float> thetas(f1.size());
	std::vector<float> rads(f1.size());
	float max = -1e10;
	float min = 1e10;
	float maxTheta(-1);
	float minTheta(361);
	for (int i = 0; i<f1.size(); i++)
	{
		float dx = f1[i].x - f2[i].x;
		float dy = f1[i].y - f2[i].y;
		float theta = atan2(dy, dx) / M_PI * 180 + 180;
		thetas[i] = theta;
		rads[i] = sqrt(dx*dx + dy*dy);

		max = rads[i] >max ? rads[i] : max;
		min = rads[i]<min ? rads[i] : min;
		maxTheta = std::max(theta, maxTheta);
		minTheta = std::min(minTheta, theta);
	}
	float radRange = max - 1e-6;
	float thetaRange = 360;
	//maxFlow = maxFlow>1.0f?maxFlow:1.0f;
	//std::cout << "Rad range " << radRange << " theta range " << thetaRange << std::endl;
	while (DistSize > 1)
	{
		binSize = DistSize * thetaSize;
		histogram.resize(binSize);
		ids.resize(binSize);
		for (int i = 0; i < binSize; i++)
			ids[i].clear();
		memset(&histogram[0], 0, sizeof(float)*binSize);



		float stepR = max / DistSize + 1e-5;
		float stepT = thetaRange / thetaSize;
		for (int i = 0; i<f1.size(); i++)
		{
			int r = (int)(rads[i] / stepR);
			int t = (int)(thetas[i] / stepT);
			r = r>DistSize - 1 ? DistSize - 1 : r;
			t = t>thetaSize - 1 ? thetaSize - 1 : t;
			int idx = t*DistSize + r;
			//std::cout<<idx<<std::endl;
			histogram[idx]++;
			ids[idx].push_back(i);

		}
		struct bin
		{
			bin(int i, int s) :id(i), size(s){};
			int id;
			int size;
			bool operator < (const bin& a)
			{
				return size > a.size;
			}
		};
		std::vector<bin> bins;
		for (int i = 0; i < histogram.size(); i++)
		{
			bins.push_back(bin(i, histogram[i]));
		}
		std::sort(bins.begin(), bins.end());
		//最大bin
		int max = ids[0].size();
		int idx(0);
		//第二大bin
		int scdMax(0);
		int scdIdx(0);
		for (int i = 1; i<ids.size(); i++)
		{
			if (ids[i].size() > max)
			{
				scdMax = max;
				scdIdx = idx;
				max = ids[i].size();
				idx = i;
			}
			else if (ids[i].size() > scdMax)
			{
				scdMax = ids[i].size();
				scdIdx = i;
			}

		}
		//显示max
		/*cv::Mat img= cv::Mat::zeros(480, 640, CV_8U);
		float N (0);
		for (int b = 0; b < bins.size() && N/f1.size() <0.5; b++)
		{
			int id = bins[b].id;
			N += bins[b].size;
			for (size_t j = 0; j < ids[id].size(); j++)
			{
				int x = f1[ids[id][j]].x + 0.5;
				int y = f1[ids[id][j]].y + 0.5;
				cv::circle(img, cv::Point(x, y), 5, cv::Scalar(255));
			}
		}
		
	
		cv::imshow("max bin points", img);
		cv::waitKey();*/
		float ratio = max*1.0 / f1.size();
		if (ratio < 0.1)
		{
			/*DrawHistogram(histogram, histogram.size(), "hist");
			cv::waitKey();*/
			if (scdMax*1.0 / max > 0.8 && (scdMax + max) *1.0 / f1.size() > 0.45)
			{
				for (size_t j = 0; j < ids[scdMax].size(); j++)
				{
					ids[max].push_back(ids[scdMax][j]);
					histogram[max]++;
				}
				break;
			}
			DistSize--;
			//thetaSize--;
		}
		else
			break;
	}
	
	
}
void OpticalFlowHistogram(std::vector<cv::Point2f>& f1, std::vector<cv::Point2f>& f2,
	std::vector<float>& histogram, std::vector<std::vector<int>>& ids, int DistSize ,int thetaSize,float thetaMin, float thetaMax)
{
	//直方图共DistSize * thetaSize个bin，其中根据光流强度分DistSize个bin，每个bin根据光流方向分thetaSize个bin
	int binSize = DistSize * thetaSize;
	histogram.resize(binSize);
	ids.resize(binSize);
	for(int i=0; i<binSize; i++)
		ids[i].clear();
	memset(&histogram[0],0,sizeof(float)*binSize);
	float max = -1e10;
	float min = 1e10;
	float maxTheta(-1);
	float minTheta(361);
	std::vector<float> thetas(f1.size());
	std::vector<float> rads(f1.size());
	for(int i =0; i<f1.size(); i++)
	{
		float dx = f1[i].x - f2[i].x;
		float dy = f1[i].y - f2[i].y;
		float theta = atan2(dy,dx)/M_PI*180 + 180;
		thetas[i] = theta;
		rads[i] = sqrt(dx*dx + dy*dy);
	
		max = rads[i] >max? rads[i] : max;
		min = rads[i]<min ? rads[i]: min;
		maxTheta = std::max(theta, maxTheta);
		minTheta = std::min(minTheta, theta);
	}
	float maxFlow = max+1e-6;
	float thetaRange = 360;
	//maxFlow = maxFlow>1.0f?maxFlow:1.0f;
	//std::cout << "Flow range " << maxFlow << " theta range " << thetaRange << std::endl;
	//maxFlow = 10;
	float stepR = maxFlow / DistSize;
	float stepT = thetaRange / thetaSize;
	for(int i=0; i<f1.size(); i++)
	{
		int r = (int)((rads[i])/stepR);
		int t = (int)((thetas[i]) / stepT);
		r = r>DistSize-1? DistSize-1:r;
		t = t>thetaSize-1? thetaSize-1:t;
		int idx = t*DistSize+r;
		//std::cout<<idx<<std::endl;
		histogram[idx]++;
		ids[idx].push_back(i);
	
	}
}
void OpticalFlowHistogram(std::vector<cv::Point2f>& f1, std::vector<cv::Point2f>& f2, std::vector<float>& rads, std::vector<float>& thetas,
	std::vector<float>& histogram, std::vector<std::vector<int>>& ids, int distSize,int thetaSize, float thetaMin, float thetaMax)

{
	
	float max = -1e10;
	float min = -max;
	thetas.resize(f1.size());
	rads.resize(f1.size());
	for(int i =0; i<f1.size(); i++)
	{
		float dx = f1[i].x - f2[i].x;
		float dy = f1[i].y - f2[i].y;
		float theta = atan2(dy,dx)/M_PI*180 + 180;
		thetas[i] = theta;
		rads[i] = sqrt(dx*dx + dy*dy);
	
		max = rads[i] >max? rads[i] : max;
		min = rads[i]<min ? rads[i]: min;

	}
	float maxFlow = max-min+1e-6;
	maxFlow = maxFlow>1.0f?maxFlow:1.0f;

	//std::cout<<"maxFlow "<<maxFlow<<std::endl;
	
	//直方图共DistSize * thetaSize个bin，其中根据光流强度分DistSize个bin，每个bin根据光流方向分thetaSize个bin
	int binSize = distSize * thetaSize;
	histogram.resize(binSize);
	ids.resize(binSize);
	for(int i=0; i<binSize; i++)
		ids[i].clear();
	memset(&histogram[0],0,sizeof(float)*binSize);
	float stepR = (maxFlow)/distSize;
	float stepT = (thetaMax - thetaMin)/thetaSize;
	for(int i=0; i<f1.size(); i++)
	{
		int r = (int)((rads[i] - min)/stepR);
		int t = (int)((thetas[i]-thetaMin)/stepT);
		r = r>distSize-1? distSize-1:r;
		t = t>thetaSize-1? thetaSize-1:t;
		int idx = t*distSize+r;
		//std::cout<<idx<<std::endl;
		histogram[idx]++;
		ids[idx].push_back(i);
	
	}
}
void OpticalFlowHistogram(std::vector<float>& rads, std::vector<float>& thetas,
	std::vector<float>& histogram, std::vector<std::vector<int>>& ids, int DistSize ,int thetaSize, float thetaMin, float thetaMax )

{
	//直方图共DistSize * thetaSize个bin，其中根据光流强度分DistSize个bin，每个bin根据光流方向分thetaSize个bin
	int binSize = DistSize * thetaSize;
	histogram.resize(binSize);
	ids.resize(binSize);
	for(int i=0; i<binSize; i++)
		ids[i].clear();
	memset(&histogram[0],0,sizeof(float)*binSize);
	float max = -9999;
	float min = -max;
	for(int i =0; i<rads.size(); i++)
	{
		
		max = rads[i] >max? rads[i] : max;
		min = rads[i]<min ? rads[i]: min;

	}
	std::cout<<"fine step maxflow "<<max-min<<std::endl;
	float stepR = (max-min+1e-6)/DistSize;
	float stepT = (thetaMax - thetaMin+1e-6)/thetaSize;
	for(int i=0; i<rads.size(); i++)
	{
		int r = (int)((rads[i] - min)/stepR);
		int t = (int)((thetas[i]-thetaMin)/stepT);
		r = r>DistSize-1? DistSize-1:r;
		t = t>thetaSize-1? thetaSize-1:t;
		int idx = t*DistSize+r;
		//std::cout<<idx<<std::endl;
		histogram[idx]++;
		ids[idx].push_back(i);
	
	}
}
float FeedbackOpticalFlowHistogram(std::vector<cv::Point2f>& f1, std::vector<cv::Point2f>& f2, std::vector<float>& rads, std::vector<float>& thetas,
	std::vector<float>& histogram, std::vector<std::vector<int>>& ids, int& distSize, int thetaSize , float thetaMin, float thetaMax)
{
	float max = -1e10;
	float min = -max;
	thetas.resize(f1.size());
	rads.resize(f1.size());
	for(int i =0; i<f1.size(); i++)
	{
		float dx = f1[i].x - f2[i].x;
		float dy = f1[i].y - f2[i].y;
		float theta = atan2(dy,dx)/M_PI*180 + 180;
		thetas[i] = theta;
		rads[i] = sqrt(dx*dx + dy*dy);
	
		max = rads[i] >max? rads[i] : max;
		min = rads[i]<min ? rads[i]: min;

	}
	float maxFlow = max-min+1e-6;
	

	//std::cout<<"maxFlow "<<maxFlow<<std::endl;
	distSize = (int)(maxFlow/distSize + 0.5);
	distSize = distSize >1 ? distSize :1;
	//直方图共DistSize * thetaSize个bin，其中根据光流强度分DistSize个bin，每个bin根据光流方向分thetaSize个bin
	int binSize = distSize * thetaSize;
	histogram.resize(binSize);
	ids.resize(binSize);
	for(int i=0; i<binSize; i++)
		ids[i].clear();
	memset(&histogram[0],0,sizeof(float)*binSize);
	float stepR = (maxFlow)/distSize;
	float stepT = (thetaMax - thetaMin)/thetaSize;
	for(int i=0; i<f1.size(); i++)
	{
		int r = (int)((rads[i] - min)/stepR);
		int t = (int)((thetas[i]-thetaMin)/stepT);
		r = r>distSize-1? distSize-1:r;
		t = t>thetaSize-1? thetaSize-1:t;
		int idx = t*distSize+r;
		//std::cout<<idx<<std::endl;
		histogram[idx]++;
		ids[idx].push_back(i);
	
	}
	return maxFlow;
}
void OpticalFlowHistogram(const cv::Mat& flow,
	std::vector<float>& histogram, std::vector<float>&avgDx, std::vector<float>& avgDy,std::vector<std::vector<int>>& ids, cv::Mat& flowIdx,int DistSize,int thetaSize)
{
	flowIdx.create(flow.size(),CV_16U);
	//直方图共256个bin，其中根据光流强度分16个bin，每个bin根据光流方向分16个bin
	//直方图共distSize和thetaSize个Bin，首先根据光流方向分，然后再根据强度分
	int binSize = DistSize * thetaSize;
	histogram.resize(binSize);
	avgDx.resize(binSize);
	avgDy.resize(binSize);
	ids.resize(binSize);
	for(int i=0; i<binSize; i++)
		ids[i].clear();
	memset(&histogram[0],0,sizeof(float)*binSize);
	memset(&avgDx[0],0,sizeof(float)*binSize);
	memset(&avgDy[0],0,sizeof(float)*binSize);

	cv::Mat xy[2];
	cv::split(flow, xy);

	//calculate angle and magnitude
	cv::Mat magnitude, angle;
	cv::cartToPolar(xy[0], xy[1], magnitude, angle, true);
	//angle range -180~180
	//translate magnitude to range [0;1]
	double mag_max;
	cv::minMaxLoc(magnitude, 0, &mag_max);
	magnitude.convertTo(magnitude, -1, 1.0/mag_max);

	
	float stepR = 1.0/DistSize;
	float stepT = 360.0/thetaSize;
	float* magPtr = (float*)magnitude.data;
	float* angPtr = (float*)angle.data;
	for(int i = 0; i<magnitude.rows; i++)
	{
		float* magPtr = magnitude.ptr<float>(i);
		float* angPtr = angle.ptr<float>(i);
		unsigned short* indPtr = flowIdx.ptr<unsigned short>(i);
		for(int j=0; j<magnitude.cols; j++)
		{
			int index = i*magnitude.cols+j;
			int r = (int)(magPtr[j]/stepR);
			int t = (int)((angPtr[j]+180)/stepT);
			//std::cout<<magnitude.at<float>(i,j)<<","<<angle.at<float>(i,j)<<std::endl;
			r = r>=DistSize? DistSize-1:r;
			t = t>=thetaSize? thetaSize-1:t;
			unsigned int idx = t*DistSize+r;
			indPtr[j] = idx;
			//std::cout<<idx<<std::endl;
			histogram[idx]++;
			avgDx[idx] +=*(float*)(xy[0].data+index*4);
			avgDy[idx] +=*(float*)(xy[1].data+index*4);
			ids[idx].push_back(index);

		}
	}
	
}
void makecolorwheel(std::vector<cv::Scalar> &colorwheel)
{

	int RY = 15;
	int YG = 6;
	int GC = 4;
	int CB = 11;
	int BM = 13;
	int MR = 6;

	int i;

	for (i = 0; i < RY; i++) colorwheel.push_back(cv::Scalar(255, 255 * i / RY, 0));
	for (i = 0; i < YG; i++) colorwheel.push_back(cv::Scalar(255 - 255 * i / YG, 255, 0));
	for (i = 0; i < GC; i++) colorwheel.push_back(cv::Scalar(0, 255, 255 * i / GC));
	for (i = 0; i < CB; i++) colorwheel.push_back(cv::Scalar(0, 255 - 255 * i / CB, 255));
	for (i = 0; i < BM; i++) colorwheel.push_back(cv::Scalar(255 * i / BM, 0, 255));
	for (i = 0; i < MR; i++) colorwheel.push_back(cv::Scalar(255, 0, 255 - 255 * i / MR));
}

void FeatureFlowColor(cv::Mat& img, std::vector<cv::Point2f>& f1, std::vector<cv::Point2f>& f2)
{
	static std::vector<cv::Scalar> colorwheel; //Scalar r,g,b  
	if (colorwheel.empty())
		makecolorwheel(colorwheel);

	float maxRad(-1);
	float minRad(1e10);
	std::vector<float> dx(f1.size()), dy(f1.size());
	for (size_t i = 0; i < f1.size(); i++)
	{
		dx[i] = f1[i].x - f2[i].x;
		dy[i] = f1[i].y - f2[i].y;
		float rad = sqrt(dx[i] * dx[i] + dy[i] * dy[i]);
		maxRad = rad>maxRad ? rad : maxRad;
		minRad = rad < minRad ? rad : minRad;
	}
	if (maxRad < 1e-5)
		maxRad = 1e-5;
	std::cout << "maxRad " << maxRad << " minRad " << minRad << " range " << maxRad - minRad << "\n";

	float maxTheta(0), minTheta(360);
	for (size_t i = 0; i < f1.size(); i++)
	{
		float fx = dx[i] / maxRad;
		float fy = dy[i] / maxRad;
		float rad = sqrt(fx * fx + fy * fy);
		float theta = atan2(-fy, -fx) / CV_PI * 180 + 180;
		maxTheta = theta>maxTheta ? theta : maxTheta;
		minTheta = theta < minTheta ? theta : minTheta;
		float angle = atan2(-fy, -fx) / CV_PI;
		float fk = (angle + 1.0) / 2.0 * (colorwheel.size() - 1);
		int k0 = (int)fk;
		int k1 = (k0 + 1) % colorwheel.size();
		float f = fk - k0;
		//f = 0;
		cv::Scalar data;
		for (int b = 0; b < 3; b++)
		{
			float col0 = colorwheel[k0][b] / 255.0;
			float col1 = colorwheel[k1][b] / 255.0;
			float col = (1 - f) * col0 + f * col1;
			if (rad <= 1)
				col = 1 - rad * (1 - col); // increase saturation with radius  
			else
				col *= .75; // out of range  
			data[2 - b] = (int)(255.0 * col);
		}
		cv::circle(img, f1[i], 3, data);
	}
	std::cout << "maxTheta " << maxTheta << " minTheta " << minTheta << " range " << maxTheta - minTheta << "\n";
	std::cout << "----------------------------------------\n";

}

void FeaturePointsRefineHistogram(std::vector<cv::Point2f>& features1, std::vector<cv::Point2f>& features2, std::vector<uchar>& inliers, int distSize, int thetaSize)
{
	inliers.resize(features1.size());
	
	for (int i = 0; i < features1.size(); i++)
	{
		inliers[i] = 0;
	}
	std::vector<float> histogram;
	std::vector<std::vector<int>> ids;

	//OpticalFlowHistogram(features1, features2, histogram, ids, distSize, thetaSize);
	IterativeOpticalFlowHistogram(features1, features2, histogram, ids, distSize, thetaSize);
	
	//std::cout << ratio << "\n";
	//最大bin
	int max = ids[0].size();
	int idx(0);
	for (int i = 1; i<ids.size(); i++)
	{
		if (ids[i].size() > max)
		{
			max = ids[i].size();
			idx = i;
		}
	}
	for (int i = 0; i<ids[idx].size(); i++)
	{
		inliers[ids[idx][i]] = 1;		
	}
	
}
void FeaturePointsRefineHistogram(int width, int height,std::vector<cv::Point2f>& features1, std::vector<cv::Point2f>& features2,int distSize, int thetaSize)
{
	std::vector<float> histogram;
	std::vector<std::vector<int>> ids;

	OpticalFlowHistogram(features1,features2,histogram,ids,distSize,thetaSize);
	
	//最大bin
	int max =ids[0].size(); 
	int idx(0);
	for(int i=1; i<ids.size(); i++)
	{
		if (ids[i].size() > max)
		{
			max = ids[i].size();
			idx = i;
		}
	}
	int k=0;
	for(int i=0; i<ids[idx].size(); i++)
	{
		features1[k] = features1[ids[idx][i]];
		features2[k] = features2[ids[idx][i]];
		k++;
	}
	
	features1.resize(k);
	features2.resize(k);
}


void OpticalFlowHistogramO(std::vector<cv::Point2f>& f1, std::vector<cv::Point2f>& f2,
	std::vector<float>& histogram, std::vector<std::vector<int>>& ids, int binSize)
{
	//	
	std::vector<float> thetas(f1.size());
	float maxTheta(0), minTheta(360);
	for (int i = 0; i<f1.size(); i++)
	{
		float dx = f1[i].x - f2[i].x;
		float dy = f1[i].y - f2[i].y;
		float theta = atan2(dy, dx) / M_PI * 180 + 180;
		thetas[i] = theta;
		maxTheta = std::max(theta, maxTheta);
		minTheta = std::min(minTheta, theta);
	}
	histogram.resize(binSize);
	ids.resize(binSize);
	for (int i = 0; i<binSize; i++)
		ids[i].clear();
	memset(&histogram[0], 0, sizeof(float)*binSize);
	
	float stepT = (maxTheta - minTheta) / binSize;
	for (int i = 0; i<f1.size(); i++)
	{
		
		int idx = (int)((thetas[i] - minTheta) / stepT);
		idx = std::min(idx, binSize - 1);
		//std::cout<<idx<<std::endl;
		histogram[idx]++;
		ids[idx].push_back(i);

	}
}
void FeaturePointsRefineHistogramO(std::vector<cv::Point2f>& features1, std::vector<cv::Point2f>& features2, int thetaSize)
{
	std::vector<uchar> inliers;
	FeaturePointsRefineHistogramO(features1, features2, inliers, thetaSize);
	int k = 0;
	for (int i = 0; i<inliers.size(); i++)
	{
		if (inliers[i] == 1)
		{
			features1[k] = features1[i];
			features2[k] = features2[i];
			k++;
		}	
	}

	features1.resize(k);
	features2.resize(k);
}
void FeaturePointsRefineHistogramO(std::vector<cv::Point2f>& features1, std::vector<cv::Point2f>& features2, std::vector<uchar>& inliers, int thetaSize)
{
	inliers.resize(features1.size());
	memset(&inliers[0], 0, inliers.size());
	std::vector<float> histogram;
	std::vector<std::vector<int>> ids;
	OpticalFlowHistogramO(features1, features2, histogram, ids, thetaSize);

	//最大bin
	int max = ids[0].size();
	int idx(0);
	for (int i = 1; i<ids.size(); i++)
	{
		if (ids[i].size() > max)
		{
			max = ids[i].size();
			idx = i;
		}
	}
	for (int i = 0; i<ids[idx].size(); i++)
	{
		inliers[ids[idx][i]] = 1;
	
	}
	
}

//求均值
template<typename T>
T average(std::vector<T>& data)
{
	T sum(0);
	for(int i=0; i<data.size(); i++)
	{
		sum+= data[i];
	}
	return sum/data.size();
}
//求方差
template<typename T>
T varition(std::vector<T>& data)
{
	T avg = average(data);
	T sum(0);
	for(int i=0; i<data.size(); i++)
	{
		sum += (data[i] - avg)*(data[i] - avg);
	}
	return sum/data.size();
}

//coarse to fine refine
void C2FFeaturePointsRefineHistogram(int width, int height,std::vector<cv::Point2f>& features1, std::vector<cv::Point2f>& features2,int radSize1, int thetaSize1,int radSize2, int thetaSize2)
{
	//首先用较粗的bin选取一部分特征点	
	std::vector<float> histogram;
	std::vector<std::vector<int>> ids;
	std::vector<float> rads,thetas;
	int distSize = radSize1;
	int thetaSize = thetaSize1;
	OpticalFlowHistogram(features1,features2,rads,thetas,histogram,ids,distSize,thetaSize);

	//最大bin
	int max =ids[0].size(); 
	int idx(0);
	for(int i=1; i<ids.size(); i++)
	{
		if (ids[i].size() > max)
		{
			max = ids[i].size();
			idx = i;
		}
	}
	
	int k=0;
	float minTheta = 361;
	float maxTheta = -1;
	float minRad = width;
	float maxRad = -1;
	for(int i=0; i<ids[idx].size(); i++)
	{
		features1[k] = features1[ids[idx][i]];
		features2[k] = features2[ids[idx][i]];
		rads[k] = rads[ids[idx][i]];
		thetas[k] = thetas[ids[idx][i]];
		if (rads[k] > maxRad)
			maxRad = rads[k];
		if (rads[k] < minRad)
			minRad = rads[k];

		if (thetas[k] < minTheta)
			minTheta = thetas[k];
		if (thetas[k] > maxTheta)
			maxTheta = thetas[k];
		k++;
	}
	features1.resize(k);
	features2.resize(k);
	thetas.resize(k);
	rads.resize(k);
	std::cout<<"after first histogram "<<k<<std::endl;
	float vRads = varition(rads);
	float vThetas = varition(thetas);
	std::cout<<"	vRads "<<vRads<<" vThetas "<<vThetas<<" \n";
	/*float minTheta = idx/distSize * 360/thetaSize;
	float maxTheta = minTheta + 360/thetaSize;*/
	std::cout<<"	minTheta "<<minTheta<<" maxTheta "<<maxTheta<<std::endl;
	std::cout<<"	minRad "<<minRad<<" maxRad "<<maxRad<<std::endl;
	if (maxTheta - minTheta < 1)
		thetaSize2 = 1;
	else 
		thetaSize2 = (int) (maxTheta - minTheta+0.5);
	if (maxRad - minRad < 1)
		radSize2 = 1;
	else
		radSize2 = (int)(maxRad - minRad +0.5);
	OpticalFlowHistogram(rads,thetas,histogram,ids,radSize2,thetaSize2,minTheta,maxTheta);
	//最大bin
	max =ids[0].size(); 
	idx = 0;
	for(int i=1; i<ids.size(); i++)
	{
		if (ids[i].size() > max)
		{
			max = ids[i].size();
			idx = i;
		}
	}
	k=0;
	for(int i=0; i<ids[idx].size(); i++)
	{
		features1[k] = features1[ids[idx][i]];
		features2[k] = features2[ids[idx][i]];
		k++;
	}
	
	features1.resize(k);
	features2.resize(k);
	
}
//分块Id，width图像宽度，height图像高度，pt输入点坐标，blockWidth 分块宽度
int blockId(int width, int height, cv::Point2f pt, int blockWidth)
{
	int blkWidth = width/blockWidth;
	int blkHeight = height/blockWidth;
	int idx = (int)(pt.x+0.5) / blkWidth;
	int idy = (int)(pt.y+0.5) / blkHeight;
	return idx + idy*blockWidth;

}

int BlockDltHomography(int width, int height, int quadWidth, std::vector<cv::Point2f>& features1, std::vector<cv::Point2f>& features0, 
	std::vector<cv::Mat>& homographies,std::vector<float>& blkWeights,
	std::vector<cv::Point2f>& sf1,std::vector<cv::Point2f>& sf0)
{
	const int minNumForHomography = 10;
	int blkWidth = width/quadWidth;
	int blkHeight = height/quadWidth;
	int blkSize = quadWidth*quadWidth;
	homographies.resize(blkSize);
	blkWeights.resize(blkSize);
	std::vector<std::vector<cv::Point2f>> f1,f0;
	f0.resize(blkSize);
	f1.resize(blkSize);
	
	for(int i=0; i< features1.size(); i++)
	{
		int idx = blockId(width,height,features1[i],quadWidth);
		f1[idx].push_back(features1[i]);
		f0[idx].push_back(features0[i]);
	}
	float num(1e-6);
	float avgErr(0);
	for(int i=0; i<blkSize; i++)
	{
		
		
		
		if (f1[i].size() >= minNumForHomography)
		{
			//std::cout<<"	block "<< i<<"  has "<<f1[i].size()<<" features, homography estimated\n";
			cv::Mat homo;
			/*std::vector<uchar> inliers;*/
			//findHomographyDLT(f1[i],f0[i],homo);
			//findHomographyNormalizedDLT(f1[i],f0[i],homo);
			findHomographyEqa(f1[i],f0[i],homo);
			//计算平均误差			
			MatrixTimesPoints(homo,f1[i]);
			float err(0);
			for(int j=0; j<f1[i].size(); j++)
			{
				float ex = f1[i][j].x-f0[i][j].x;
				float ey = f1[i][j].y - f0[i][j].y;
				err += sqrt(ex*ex+ey*ey);

			}
			err/=f1[i].size();
			//std::cout<<"error of cell "<<i<<" "<<err<<std::endl;
			/*blkWeights[i] = exp((float)-blkSize*f1[i].size()/features1.size())*3;
			blkWeights[i] = blkWeights[i]<0.3 ? 0.3 : blkWeights[i];*/
			/*if (err > 0.06)
			{
				blkWeights[i] = 0.8;

			}
			else if (err<0.02)
			{
				blkWeights[i] = 0.5;
			}
			else
			{
				blkWeights[i] = err;
			}*/
			blkWeights[i] = err;
			//homo = cv::findHomography(f1[i],f0[i],inliers,CV_RANSAC,0.1);
			//std::cout<<homo<<"\n";
			homographies[i] = homo.clone();
			avgErr += err;
			num++;
		}	
		else
		{
			blkWeights[i] = 1.0;
			/*if (f1[i].size() > 0)
				std::cout<<"	block "<< i<<"  has "<<f1[i].size()<<" features Blinear term added\n";*/
			for(int j=0; j< f1[i].size(); j++)
			{
				sf1.push_back(f1[i][j]);
				sf0.push_back(f0[i][j]);
			}
		
		}

		
	}
	avgErr /= num;
	//std::cout<<"avg error "<<avgErr<<std::endl;
	for(int i=0; i< blkWeights.size(); i++)
	{
		blkWeights[i] = exp(blkWeights[i] / avgErr)*0.15;
		if (blkWeights[i] > 1.0)
			blkWeights[i] = 1.0;
		else if(blkWeights[i] < 0.3)
			blkWeights[i] = 0.3;
		/*if (blkWeights[i] > 10)
			blkWeights[i] = 1.0;
		else if(blkWeights[i] > 1)
			blkWeights[i] = 0.8;
		else if(blkWeights[i] < 0.5)
			blkWeights[i] = 0.5;*/
	}
	return num;
}
//分块背景特征点直方图求精
void BC2FFeaturePointsRefineHistogram(int width, int height,std::vector<cv::Point2f>& features1, std::vector<cv::Point2f>& features2, std::vector<float>& blkWeights, int quadWidth,int radSize1, int thetaSize1, int radSize2, int thetaSize2)
{
	//首先用较粗的bin选取一部分特征点	
	std::vector<float> histogram;
	std::vector<std::vector<int>> ids;
	std::vector<float> rads,thetas;
	int distSize = radSize1;
	int thetaSize = thetaSize1;
	float maxFlow = FeedbackOpticalFlowHistogram(features1,features2,rads,thetas,histogram,ids,distSize,thetaSize);

	//最大bin
	int max =ids[0].size(); 
	int idx(0);
	for(int i=1; i<ids.size(); i++)
	{
		if (ids[i].size() > max)
		{
			max = ids[i].size();
			idx = i;
		}
	}
	float minTheta = idx/distSize * 360/thetaSize;
	float maxTheta = minTheta + 360/thetaSize;
	float minRad = idx%distSize * maxFlow/distSize;
	float maxRad = minRad + maxFlow/distSize;
	std::vector<std::vector<cv::Point2f>> blockF1;
	std::vector<std::vector<cv::Point2f>> blockF2;
	std::vector<std::vector<float>> blockRads;
	std::vector<std::vector<float>> blockThetas;
	int blockNum = quadWidth * quadWidth;
	blockF1.resize(blockNum);
	blockF2.resize(blockNum);
	blockRads.resize(blockNum);
	blockThetas.resize(blockNum);
	blkWeights.resize(blockNum);
	memset(&blkWeights[0],0,sizeof(int)*blockNum);
	for(int i=0; i<blockNum; i++)
	{
		blockF1[i].clear();
		blockF2[i].clear();
		blockRads[i].clear();
		blockThetas[i].clear();
	}
	int k=0;
	for(int i=0; i<ids[idx].size(); i++)
	{
		features1[k] = features1[ids[idx][i]];		
		features2[k] = features2[ids[idx][i]];		
		rads[k] = rads[ids[idx][i]];
		thetas[k] = thetas[ids[idx][i]];
		int blkId = blockId(width,height,features1[k],quadWidth);
		blockF1[blkId] .push_back(features1[k]);
		blockF2[blkId].push_back(features2[k]);
		blockRads[blkId].push_back(rads[k]);
		blockThetas[blkId].push_back(thetas[k]);
		k++;
	}
	
	k=0;
	for(int i=0; i<blockF1.size(); i++)
	{
		if (blockF1[i].size() == 0)
			continue;
		OpticalFlowHistogram(blockRads[i],blockThetas[i],histogram,ids,radSize2,thetaSize2,minTheta,maxTheta);
		int max =ids[0].size(); 
		int idx(0);
		for(int i=1; i<ids.size(); i++)
		{
			if (ids[i].size() > max)
			{
				max = ids[i].size();
				idx = i;
			}
		}
		blkWeights[i] = ids[idx].size();
		for(int m=0; m<ids[idx].size(); m++)
		{
			features1[k] = blockF1[i][ids[idx][m]];		
			features2[k] = blockF2[i][ids[idx][m]];		
			k++;

		}
	}
	
	
	features1.resize(k);
	features2.resize(k);
	for(int i=0; i<blockNum; i++)
	{
		blkWeights[i] = exp(-blockNum*blkWeights[i]/k)*3;
		blkWeights[i] = blkWeights[i]<0.3 ? 0.3 : blkWeights[i];
	}
}
//nf是特征点数量，vf1中前面nf个是特征点，后面是超像素中心
void FeaturePointsRefineHistogram(int& nf, int width, int height,std::vector<cv::Point2f>& features1, std::vector<cv::Point2f>& features2, int distSize, int thetaSize)
{
	std::vector<float> histogram;
	std::vector<std::vector<int>> ids;
	
	
	IterativeOpticalFlowHistogram(features1,features2,histogram,ids,distSize,thetaSize);

	//最大bin
	int max =ids[0].size(); 
	int idx(0);
	for(int i=1; i<ids.size(); i++)
	{
		if (ids[i].size() > max)
		{
			max = ids[i].size();
			idx = i;
		}
	}
	int k=0;
	int nnf(0);
	for(int i=0; i<ids[idx].size(); i++)
	{
		features1[k] = features1[ids[idx][i]];
		features2[k] = features2[ids[idx][i]];
		if (ids[idx][i]<nf)
			nnf++;
		k++;
	}
	nf = nnf;
	features1.resize(k);
	features2.resize(k);
}

void KLTFeaturesMatching(const cv::Mat& simg, const cv::Mat& timg, std::vector<cv::Point2f>& vf1, std::vector<cv::Point2f>& vf2, int cornerCount, float quality, float minDist)
{
	vf1.clear();
	vf2.clear();
	std::vector<uchar>status;
	std::vector<float> err;
	cv::Mat sGray,tGray;
	if (simg.channels() == 3)
		cv::cvtColor(simg,sGray,CV_BGR2GRAY);
	else
		sGray = simg;
	if (timg.channels() == 3)
		cv::cvtColor(timg,tGray,CV_BGR2GRAY);
	else
		tGray = timg;
	cv::goodFeaturesToTrack(sGray,vf1,cornerCount,quality,minDist);
	cv::calcOpticalFlowPyrLK(sGray,tGray,vf1,vf2,status,err);
	int k=0;
	for(int i=0; i<vf1.size(); i++)
	{
		if(status[i] == 1)
		{
			vf1[k] = vf1[i];
			vf2[k] = vf2[i];
			k++;
		}
	}

	vf1.resize(k);
	vf2.resize(k);
	//FeaturePointsRefineHistogram(vf1,vf2);
}
void FILESURFFeaturesMatching(const cv::Mat& simg, const cv::Mat& timg, std::vector<cv::Point2f>& vf1, std::vector<cv::Point2f>& vf2)
{
	FILE* f1 = fopen("f1.txt","r");
	FILE* f2 = fopen("f2.txt","r");	
	float x,y;
	while(fscanf(f1,"%f\t%f",&x,&y) > 0)
	{
		
		vf1.push_back(cv::Point2f(x-1,y-1));
		
	}
	while(fscanf(f2,"%f\t%f",&x,&y)>0)
	{
		
		vf2.push_back(cv::Point2f(x-1,y-1));
		
	}
	fclose(f1);
	fclose(f2);
}
void SURFFeaturesMatching(const cv::Mat& simg, const cv::Mat& timg, std::vector<cv::Point2f>& vf1, std::vector<cv::Point2f>& vf2)
{
	using namespace cv;
	Mat img_1,img_2;
	if (simg.channels() == 3)
		cv::cvtColor(simg,img_1,CV_BGR2GRAY);
	else
		img_1 = simg;
	if (timg.channels() == 3)
		cv::cvtColor(timg,img_2,CV_BGR2GRAY);
	else
		img_2 = timg;

	
	//-- Step 1: Detect the keypoints using SURF Detector
    int minHessian = 400;
  
    SurfFeatureDetector detector( minHessian );
  
    std::vector<KeyPoint> keypoints_object, keypoints_scene;
  
    detector.detect( img_1, keypoints_object );
    detector.detect( img_2, keypoints_scene );
  
    //-- Step 2: Calculate descriptors (feature vectors)
    SurfDescriptorExtractor extractor;
  
    Mat descriptors_object, descriptors_scene;
  
    extractor.compute( img_1, keypoints_object, descriptors_object );
    extractor.compute( img_2, keypoints_scene, descriptors_scene );
  
    //-- Step 3: Matching descriptor vectors using FLANN matcher
    FlannBasedMatcher matcher;
    std::vector< DMatch > matches;
    matcher.match( descriptors_object, descriptors_scene, matches );
  
    double max_dist = 0; double min_dist = 100;
  
    //-- Quick calculation of max and min distances between keypoints
    for( int i = 0; i < descriptors_object.rows; i++ )
    { double dist = matches[i].distance;
      if( dist < min_dist ) min_dist = dist;
      if( dist > max_dist ) max_dist = dist;
    }
  
    printf("-- Max dist : %f \n", max_dist );
    printf("-- Min dist : %f \n", min_dist );
  
    //-- Draw only "good" matches (i.e. whose distance is less than 3*min_dist )
    std::vector< DMatch > good_matches;
  
    for( int i = 0; i < descriptors_object.rows; i++ )
    {
		if( matches[i].distance < 3*min_dist + 1e-2)
       { 
		   good_matches.push_back( matches[i]); 
		}
    }
  
   
  
	vf1.clear();
	vf2.clear();
    for( int i = 0; i < good_matches.size(); i++ )
    {
      //-- Get the keypoints from the good matches
      vf1.push_back( keypoints_object[ good_matches[i].queryIdx ].pt );
      vf2.push_back( keypoints_scene[ good_matches[i].trainIdx ].pt );
    }
}

void MatchingResult(const cv::Mat& simg, const cv::Mat& timg, const std::vector<cv::Point2f>& features1, const std::vector<cv::Point2f>& features2,cv::Mat& matchingRst)
{
	matchingRst.create(simg.rows,simg.cols*2,simg.type());
	simg.copyTo(matchingRst(cv::Rect(0,0,simg.cols,simg.rows)));
	timg.copyTo(matchingRst(cv::Rect(simg.cols,0,simg.cols,simg.rows)));
	cv::Scalar color(255,0,0);
	cv::Scalar color2(0,255,0);
	for(int i=0; i< features1.size(); i++)
	{
		cv::Point2f pt1 = features1[i];
		cv::Point2f pt2 = cv::Point2f(features2[i].x+simg.cols,features2[i].y);
		cv::circle(matchingRst,pt1,3,color);
		cv::circle(matchingRst,pt2,3,color);
		cv::line(matchingRst,pt1,pt2,color2);
	}

}


void RelFlowRefine(std::vector<cv::Point2f>& features1, std::vector<cv::Point2f>&features0, std::vector<uchar>& inliers, int& ankorId, float threshold)
{
	//select a anchor
	double minDistance(1e10);
	int anchor = 0;
	std::vector<float> dx0, dy0, dx1, dy1;
	dx0.resize(features0.size());
	dy0.resize(dx0.size());
	dx1.resize(dx0.size());
	dy1.resize(dy0.size());
	for (int a = 0; a < features0.size(); a++)
	{
		cv::Point2f anchorPt1 = features1[a];
		cv::Point2f anchorPt0 = features0[a];

		for (int i = 0; i < features1.size(); i++)
		{
			dx1[i] = features1[i].x - anchorPt1.x;
			dy1[i] = features1[i].y - anchorPt1.y;
		}

		double sum(0);
		double maxD(0);

		for (int i = 0; i < features0.size(); i++)
		{
			double dx0 = features0[i].x - anchorPt0.x;
			double dy0 = features0[i].y - anchorPt0.y;
			double diffX = dx0 - dx1[i];
			double diffY = dy0 - dy1[i];
			double diff = sqrt(diffX*diffX + diffY*diffY);
			sum += diff;
		}
		if (sum < minDistance)
		{
			minDistance = sum;
			anchor = a;
		}
	}
	ankorId = anchor;
	cv::Point2f anchorPt1 = features1[anchor];
	cv::Point2f anchorPt0 = features0[anchor];

	for (int i = 0; i < features1.size(); i++)
	{

		dx1[i] = features1[i].x - anchorPt1.x;
		dy1[i] = features1[i].y - anchorPt1.y;
	}
	double t = threshold;
	if (t < 0)
		t = minDistance / (features0.size() - 1);
	/*std::cout << "threshold = " << t << "\n";*/
	
	inliers.resize(features0.size());
	inliers[anchor] = 1;
	for (int i = 0; i < features0.size(); i++)
	{

		double dx0 = features0[i].x - anchorPt0.x;
		double dy0 = features0[i].y - anchorPt0.y;
		double diffX = dx0 - dx1[i];
		double diffY = dy0 - dy1[i];
		double diff = sqrt(diffX*diffX + diffY*diffY);

		if (diff > t)
			inliers[i] = 0;
		else
			inliers[i] = 1;


	}


}

void ShowFeatureRefine(cv::Mat& img1, std::vector<cv::Point2f>& features1, cv::Mat& img0, std::vector<cv::Point2f>&features0, std::vector<uchar>& inliers, std::string title, bool line)
{
	int width = img1.cols;
	int height = img1.rows;
	cv::Size size2(width * 2, height);
	cv::Mat rstImg(size2, CV_8UC3);
	img1.copyTo(rstImg(cv::Rect(0, 0, width, height)));
	img0.copyTo(rstImg(cv::Rect(width, 0, width, height)));
	cv::Scalar blue = cv::Scalar(255, 0, 0);
	cv::Scalar red = cv::Scalar(0, 0, 255);
	float inlierNum(0.f);
	for (int i = 0; i < inliers.size(); i++)
	{
		if (inliers[i] == 1)
		{
			cv::circle(rstImg, features1[i], 3, red);
			cv::circle(rstImg, cv::Point(features0[i].x + width, features0[i].y), 3, red);
			if (line)
				cv::line(rstImg, cv::Point(features1[i].x, features1[i].y), cv::Point(features0[i].x + width, features0[i].y), red);
			inlierNum++;
		}
		else
		{
			cv::circle(rstImg, features1[i], 3, blue);
			cv::circle(rstImg, cv::Point(features0[i].x + width, features0[i].y), 3, blue);
			if (line)
				cv::line(rstImg, cv::Point(features1[i].x, features1[i].y), cv::Point(features0[i].x + width, features0[i].y), blue);
		}
		

	}
	if (title[title.size() - 4] == '.')
	{
		cv::imwrite(title, rstImg);
	}
	else
	{
		//cv::imshow("Feature Refine, Red: Possible background; Blue: Possible foreground", rstImg);
		cv::imshow(title, rstImg);
		cv::waitKey(0);
	}
	//std::cout << "inlier pct " << inlierNum / inliers.size() << "\n";
}

void ShowFeatureRefine(cv::Mat& img1, std::vector<cv::Point2f>& features1, cv::Mat& img0, std::vector<cv::Point2f>&features0, std::vector<uchar>& inliers, std::string title, int anchorId)
{
	int width = img1.cols;
	int height = img1.rows;
	cv::Size size2(width * 2, height);
	cv::Mat rstImg(size2, CV_8UC3);
	img1.copyTo(rstImg(cv::Rect(0, 0, width, height)));
	img0.copyTo(rstImg(cv::Rect(width, 0, width, height)));
	cv::Scalar blue = cv::Scalar(255, 0, 0);
	cv::Scalar red = cv::Scalar(0, 0, 255);
	float inlierNum(0.f);
	for (int i = 0; i < inliers.size(); i++)
	{
		if (inliers[i] == 1)
		{
			cv::circle(rstImg, features1[i], 3, red);
			cv::circle(rstImg, cv::Point(features0[i].x + width, features0[i].y), 3, red);
		/*	cv::line(rstImg, cv::Point(features1[i].x, features1[i].y),
				cv::Point(features0[i].x + width, features0[i].y), cv::Scalar(255, 0, 0));
			inlierNum++;*/
		}
		else
		{
			cv::circle(rstImg, features1[i], 3, blue);
			cv::circle(rstImg, cv::Point(features0[i].x + width, features0[i].y), 3, blue);
		}
		

	}
	cv::line(rstImg, cv::Point(features1[anchorId].x, features1[anchorId].y),
		cv::Point(features0[anchorId].x + width, features0[anchorId].y), cv::Scalar(255, 0, 0));
	if (title[title.size() - 4] == '.')
	{
		cv::imwrite(title, rstImg);
	}
	else
	{
		//cv::imshow("Feature Refine, Red: Possible background; Blue: Possible foreground", rstImg);
		cv::imshow(title, rstImg);
		cv::waitKey(0);
	}
	//std::cout << "inlier pct " << inlierNum / inliers.size() << "\n";
}

void BlockRelFlowRefine::Init()
{
	_blkWidth = _width / _quad;
	_blkHeight = _height / _quad;
	_cells.resize(_quad*_quad);
	for (size_t i = 0; i < _quad; i++)
	{
		for (size_t j = 0; j < _quad; j++)
		{
			_cells[i*_quad + j].idx = i*_quad + j;
		}
	}
}
void BlockRelFlowRefine::Refine(int id, Points& features1, Points& features0, std::vector<uchar>& inliers, int& aId)
{
	inliers.resize(features0.size());
	memset(&inliers[0], 0, sizeof(uchar)*inliers.size());
	//assign features to cells
	for (size_t i = 0; i < features1.size(); i++)
	{
		cv::Point2f pt = features1[i];
		int idx = (int)(pt.x + 0.5) / _blkWidth;
		int idy = (int)(pt.y + 0.5) / _blkHeight;
		_cells[idx + idy*_quad].featureIds.push_back(i);
	}
	int n4x[] = { -1, 0, 1, 0 };
	int n4y[] = { 0, -1, 0, 1 };

	for (size_t i = 0; i < _quad; i++)
	{
		for (size_t j = 0; j < _quad; j++)
		{
			int idx = i*_quad + j;
			if (idx != id)
				continue;
			float minDistance(1e10);
			int minId(-1);
			for (size_t n = 0; n < 4; n++)
			{
				int ny = n4y[n] + i;
				int nx = n4x[n] + j;
				if (ny >= 0 && ny < _quad && nx >= 0 && nx < _quad)
				{
					int nidx = ny*_quad + nx;

					for (size_t f = 0; f < _cells[nidx].featureIds.size(); f++)
					{
						size_t id = _cells[nidx].featureIds[f];
						double nx1 = features1[id].x;
						double ny1 = features1[id].y;
						double nx0 = features0[id].x;
						double ny0 = features0[id].y;
						double sumD(0);
						for (size_t c = 0; c < _cells[idx].featureIds.size(); c++)
						{
							int cid = _cells[idx].featureIds[c];
							double x1 = features1[cid].x;
							double y1 = features1[cid].y;
							double x0 = features0[cid].x;
							double y0 = features0[cid].y;
							double dx1 = x1 - nx1;
							double dy1 = y1 - ny1;
							double dx0 = x0 - nx0;
							double dy0 = y0 - ny0;
							double dx = dx1 - dx0;
							double dy = dy1 - dy0;
							sumD += sqrtf(dx*dx + dy*dy);
						}
						if (sumD < minDistance)
						{
							minDistance = sumD;
							minId = id;
						}

					}

				}
			}
			aId = minId;
			double maxD(0);
			for (size_t c = 0; c < _cells[idx].featureIds.size(); c++)
			{
				int cid = _cells[idx].featureIds[c];
				double nx1 = features1[minId].x;
				double ny1 = features1[minId].y;
				double nx0 = features0[minId].x;
				double ny0 = features0[minId].y;
				double x1 = features1[cid].x;
				double y1 = features1[cid].y;
				double x0 = features0[cid].x;
				double y0 = features0[cid].y;
				double dx1 = x1 - nx1;
				double dy1 = y1 - ny1;
				double dx0 = x0 - nx0;
				double dy0 = y0 - ny0;
				double dx = dx1 - dx0;
				double dy = dy1 - dy0;
				if (sqrtf(dx*dx + dy*dy) > maxD)
					maxD = sqrtf(dx*dx + dy*dy);
				if (sqrtf(dx*dx + dy*dy) < _threshold)
				{
					inliers[cid] = 1;
				}
			}
			std::cout << "max distance " << maxD << "\n";
		}
	}
}
void BlockRelFlowRefine::Refine(Points& features1, Points& features0, std::vector<uchar>& inliers)
{
	inliers.resize(features0.size());
	memset(&inliers[0], 0, sizeof(uchar)*inliers.size());
	//assign features to cells
	for (size_t i = 0; i < features1.size(); i++)
	{
		cv::Point2f pt = features1[i];
		int idx = (int)(pt.x + 0.5) / _blkWidth;
		int idy = (int)(pt.y + 0.5) / _blkHeight;
		_cells[idx + idy*_quad].featureIds.push_back(i);
	}
	int n4x[] = { -1, 0, 1, 0 };
	int n4y[] = { 0, -1, 0, 1 };
	//iterate each block, fine anchor from neighbors
	for (size_t i = 0; i < _quad; i++)
	{
		for (size_t j = 0; j < _quad; j++)
		{
			int idx = i*_quad + j;
			float minDistance(1e10);
			int minId(-1);
			for (size_t n = 0; n < 4; n++)
			{
				int ny = n4y[n] + i;
				int nx = n4x[n] + j;
				if (ny >= 0 && ny < _quad && nx >= 0 && nx < _quad)
				{
					int nidx = ny*_quad + nx;
					
					for (size_t f = 0; f < _cells[nidx].featureIds.size(); f++)
					{
						size_t id = _cells[nidx].featureIds[f];
						double nx1 = features1[id].x;
						double ny1 = features1[id].y;
						double nx0 = features0[id].x;
						double ny0 = features0[id].y;
						double sumD(0);
						for (size_t c = 0; c < _cells[idx].featureIds.size(); c++)
						{ 
							int cid = _cells[idx].featureIds[c];
							double x1 = features1[cid].x;
							double y1 = features1[cid].y;
							double x0 = features0[cid].x;
							double y0 = features0[cid].y;
							double dx1 = x1 - nx1;
							double dy1 = y1 - ny1;
							double dx0 = x0 - nx0;
							double dy0 = y0 - ny0;
							double dx = dx1 - dx0;
							double dy = dy1 - dy0;
							sumD += sqrtf(dx*dx + dy*dy);
						}
						if (sumD < minDistance)
						{
							minDistance = sumD;
							minId = id;
						}
							
					}

				}
			}
			for (size_t c = 0; c < _cells[idx].featureIds.size(); c++)
			{
				int cid = _cells[idx].featureIds[c];
				double nx1 = features1[minId].x;
				double ny1 = features1[minId].y;
				double nx0 = features0[minId].x;
				double ny0 = features0[minId].y;
				double x1 = features1[cid].x;
				double y1 = features1[cid].y;
				double x0 = features0[cid].x;
				double y0 = features0[cid].y;
				double dx1 = x1 - nx1;
				double dy1 = y1 - ny1;
				double dx0 = x0 - nx0;
				double dy0 = y0 - ny0;
				double dx = dx1 - dx0;
				double dy = dy1 - dy0;
				if (sqrtf(dx*dx + dy*dy) < _threshold)
				{
					inliers[cid] = 1;
				}
			}
		}
	}
	
}

void FeaturePointsRefineHistogram(std::vector<cv::Point2f>& features1, std::vector<cv::Point2f>& features2, int distSize, int thetaSize)
{
	std::vector<uchar> inliers;
	FeaturePointsRefineHistogram(features1, features2, inliers, distSize, thetaSize);
	int k = 0;
	for (int i = 0; i<inliers.size(); i++)
	{
		if (inliers[i] == 1)
		{
			features1[k] = features1[i];
			features2[k] = features2[i];
			k++;
		}
	}

	features1.resize(k);
	features2.resize(k);
}

void ShowFeatureRefineSingle(cv::Mat& img1, std::vector<cv::Point2f>& features1, cv::Mat& img0, std::vector<cv::Point2f>&features0, std::vector<uchar>& inliers, std::string title)
{
	cv::Mat rstMat = img1.clone();
	cv::Scalar red(0, 0, 255);
	cv::Scalar blue(255, 0, 0);
	for (size_t i = 0; i < inliers.size(); i++)
	{
		if (inliers[i] == 1)
		{
			cv::circle(rstMat, features1[i], 3, red);
			cv::line(rstMat, features1[i], features0[i], red);
		}
		else
		{
			cv::circle(rstMat, features1[i], 3, blue);
			cv::line(rstMat, features1[i], features0[i], blue);
		}
	}
	if (title[title.size() - 4] == '.')
	{
		cv::imwrite(title, rstMat);
	}
	else
	{
		//cv::imshow("Feature Refine, Red: Possible background; Blue: Possible foreground", rstImg);
		cv::imshow(title, rstMat);
		cv::waitKey(0);
	}
}

void FeaturePointsRefineZoom(int width, int height, std::vector<cv::Point2f>& features1, std::vector<cv::Point2f>& features2, std::vector<uchar>& inliers, int binSize)
{
	double ox = width / 2;
	double oy = height / 2;
	double minK(1e10);
	double maxK(-1e10);
	inliers.resize(features1.size());
	memset(&inliers[0], 0, inliers.size());
	std::vector<double> kvec(features1.size());
	std::vector<double> histogram(binSize);
	std::vector<std::vector<int>> ids(binSize);
	memset(&histogram[0], 0, sizeof(double)*binSize);
	//build histogram
	for (size_t i = 0; i < features1.size(); i++)
	{
		double dx = features1[i].x - ox;
		double dy = features1[i].y - oy;
		double d1 = sqrt(dx*dx + dy*dy);
		dx = features2[i].x - ox;
		dy = features2[i].y - oy;
		double d2 = sqrt(dx*dx + dy*dy);
		double k = d1 / d2;
		if (k > maxK)
			maxK = k;
		if (k < minK)
			minK = k;
		kvec[i] = k;
	}
	double step = (maxK - minK) / binSize + 1e-3;
	for (size_t i = 0; i < features1.size(); i++)
	{
		int id = (kvec[i]-minK) / step;
		id = std::min(binSize - 1, id);
		ids[id].push_back(i);
		histogram[id]++;
	}
	double maxV, minV;
	int maxId, minId;
	minMaxLoc(histogram, maxV, minV, maxId, minId);
	for (size_t i = 0; i < ids[maxId].size(); i++)
	{
		inliers[ids[maxId][i]] = 1;
	}
}

void FeaturePointsRefineZoom(int width, int height, std::vector<cv::Point2f>& features1, std::vector<cv::Point2f>& features2, int binSize)
{
	std::vector<uchar> inliers;
	FeaturePointsRefineZoom(width, height, features1, features2, inliers, binSize);
	int k(0);
	
	for (int i = 0; i<features1.size(); i++)
	{
		if (inliers[i] == 1)
		{
			features1[k] = features1[i];
			features2[k] = features2[i];
			k++;
		}
	}

	features1.resize(k);
	features2.resize(k);
}
