#include "FeaturePointRefine.h"

void FeaturePointsRefineRANSAC(std::vector<cv::Point2f>& vf1, std::vector<cv::Point2f>& vf2,cv::Mat& homography)
{
	std::vector<uchar> inliers(vf1.size());
	homography = cv::findHomography(
		cv::Mat(vf1), // corresponding
		cv::Mat(vf2), // points
		inliers, // outputted inliers matches
		CV_RANSAC, // RANSAC method
		0.1); // max distance to reprojection point
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
void FeaturePointsRefineRANSAC(int& nf, std::vector<cv::Point2f>& vf1, std::vector<cv::Point2f>& vf2,cv::Mat& homography)
{
	std::vector<uchar> inliers(vf1.size());

	homography = cv::findHomography(
		cv::Mat(vf1), // corresponding
		cv::Mat(vf2), // points
		inliers, // outputted inliers matches
		CV_RANSAC, // RANSAC method
		0.1); // max distance to reprojection point
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
void OpticalFlowHistogram(std::vector<cv::Point2f>& f1, std::vector<cv::Point2f>& f2,
	std::vector<float>& histogram, std::vector<std::vector<int>>& ids, int DistSize ,int thetaSize)
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

	}
	float stepR = (max-min+1e-6)/DistSize;
	float stepT = 360/thetaSize;
	for(int i=0; i<f1.size(); i++)
	{
		int r = (int)((rads[i] - min)/stepR);
		int t = (int)(thetas[i]/stepT);
		r = r>DistSize-1? DistSize-1:r;
		t = t>thetaSize-1? thetaSize-1:t;
		int idx = t*DistSize+r;
		//std::cout<<idx<<std::endl;
		histogram[idx]++;
		ids[idx].push_back(i);
	
	}
}

void OpticalFlowHistogram(const cv::Mat& flow,
	std::vector<float>& histogram, std::vector<float>&avgDx, std::vector<float>& avgDy,std::vector<std::vector<int>>& ids, cv::Mat& flowIdx,int DistSize,int thetaSize)
{
	flowIdx.create(flow.size(),CV_16U);
	//直方图共256个bin，其中根据光流强度分16个bin，每个bin根据光流方向分16个bin
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
			int t = (int)(angPtr[j]/stepT);
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
void FeaturePointsRefineHistogram(int width, int height,std::vector<cv::Point2f>& features1, std::vector<cv::Point2f>& features2)
{
	std::vector<float> histogram;
	std::vector<std::vector<int>> ids;
	int len = sqrtf(width*width + height*height);
	int distSize = len/50;
	int thetaSize = 50;
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
//nf是特征点数量，vf1中前面nf个是特征点，后面是超像素中心
void FeaturePointsRefineHistogram(int& nf, int width, int height,std::vector<cv::Point2f>& features1, std::vector<cv::Point2f>& features2)
{
	std::vector<float> histogram;
	std::vector<std::vector<int>> ids;
	int len = sqrtf(width*width + height*height);
	int distSize = len/50;
	int thetaSize = 50;
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
void KLTFeaturesMatching(const cv::Mat& simg, const cv::Mat& timg, std::vector<cv::Point2f>& vf1, std::vector<cv::Point2f>& vf2)
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
	cv::goodFeaturesToTrack(sGray,vf1,100,0.08,10);
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


