#include "ASAPWarping.h"
#include "SparseSolver.h"
#include <fstream>
bool isPointInTriangular(const cv::Point2f& pt, const cv::Point2f& V0, const cv::Point2f& V1, const cv::Point2f& V2)
{   
	float lambda1 = ((V1.y-V2.y)*(pt.x-V2.x) + (V2.x-V1.x)*(pt.y-V2.y)) / ((V1.y-V2.y)*(V0.x-V2.x) + (V2.x-V1.x)*(V0.y-V2.y));
	float lambda2 = ((V2.y-V0.y)*(pt.x-V2.x) + (V0.x-V2.x)*(pt.y-V2.y)) / ((V2.y-V0.y)*(V1.x-V2.x) + (V0.x-V2.x)*(V1.y-V2.y));
	float lambda3 = 1-lambda1-lambda2;
	if (lambda1 >= 0.0 && lambda1 <= 1.0 && lambda2 >= 0.0 && lambda2 <= 1.0 && lambda3 >= 0.0 && lambda3 <= 1.0)
		return true;
	else
		return false;

}
void ASAPWarping::CreateSmoothCons(float weight)
{
	       _rowCount = 0;
           int i=0, j=0;
           addCoefficient_5(i,j,weight);
           addCoefficient_6(i,j,weight);
           
           i=0;j=_width-1;
           addCoefficient_7(i,j,weight);
           addCoefficient_8(i,j,weight);
          
           i=_height-1;j=0;
           addCoefficient_3(i,j,weight);
           addCoefficient_4(i,j,weight);
           
           i=_height-1;j=_width-1;
           addCoefficient_1(i,j,weight);
           addCoefficient_2(i,j,weight);
           
           i=0;
           for(j=1; j<=_width-2; j++)
		   {
              addCoefficient_5(i,j,weight);
              addCoefficient_6(i,j,weight);
              addCoefficient_7(i,j,weight);
              addCoefficient_8(i,j,weight);
		   }
           
           i=_height-1;
           for(j=1; j<=_width-2; j++)
           {
			  addCoefficient_1(i,j,weight);
              addCoefficient_2(i,j,weight);
              addCoefficient_3(i,j,weight);
              addCoefficient_4(i,j,weight);
		   }
           
           j=0;
           for( i=1; i<=_height-2; i++)
		   {
              addCoefficient_3(i,j,weight);
              addCoefficient_4(i,j,weight);
              addCoefficient_5(i,j,weight);
              addCoefficient_6(i,j,weight);
		   }
           
           j=_width-1;
            for( i=1; i<=_height-2; i++)
		   {
              addCoefficient_1(i,j,weight);
              addCoefficient_2(i,j,weight);
              addCoefficient_7(i,j,weight);
              addCoefficient_8(i,j,weight);
			}
           
           for (i=1; i<= _height-2; i++)
		   {
              for (j=1; j<= _width-2;j++)
			  {
              addCoefficient_1(i,j,weight);
              addCoefficient_2(i,j,weight);
              addCoefficient_3(i,j,weight);
              addCoefficient_4(i,j,weight);
              addCoefficient_5(i,j,weight);
              addCoefficient_6(i,j,weight);
              addCoefficient_7(i,j,weight);
              addCoefficient_8(i,j,weight);
			  }
		   }
		   _sRowCount = _rowCount;
}

void ASAPWarping::SetControlPts(std::vector<cv::Point2f>& inputsPts, std::vector<cv::Point2f>& outputsPts)
{
	int len = inputsPts.size();
	_dataterm_element_orgPt = inputsPts;
	_dataterm_element_desPt = outputsPts;

	_dataterm_element_i = std::vector<float>(len,0);
	_dataterm_element_j = std::vector<float>(len,0);

	_dataterm_element_V00 = std::vector<float>(len,0);
	_dataterm_element_V01 = std::vector<float>(len,0);
	_dataterm_element_V10 = std::vector<float>(len,0);
	_dataterm_element_V11 = std::vector<float>(len,0);

	std::vector<float> coefficients;
	for(int i=0; i<len; i++)
	{
		//std::cout<<i<<std::endl;
		cv::Point2f pt = inputsPts[i];
		_dataterm_element_i[i] = (int)((ceil(pt.y)+_quadHeight-1)/(_quadHeight));
		_dataterm_element_j[i] = (int)((ceil(pt.x)+_quadWidth-1)/(_quadWidth));

		Quad qd = _source->getQuad(_dataterm_element_i[i],_dataterm_element_j[i]);


		
		qd.getBilinearCoordinates(pt,coefficients);
		_dataterm_element_V00[i] = coefficients[0];
		_dataterm_element_V01[i] = coefficients[1];
		_dataterm_element_V10[i] = coefficients[2];
		_dataterm_element_V11[i] = coefficients[3];
	}

}

void ASAPWarping::CreateDataCons(cv::Mat& b)
{
	int len = _dataterm_element_i.size();
	_num_data_cons = len*2;
	_DataConstraints.create(_num_data_cons*4,3,CV_32F);
	b.create(_num_data_cons+_num_smooth_cons,1,CV_32F);
	b = cv::Scalar(0);
	_DCc = 0;
	for(int k=0; k<len; k++)
	{
		//std::cout<<k<<std::endl;
		float i = _dataterm_element_i[k];
		float j = _dataterm_element_j[k];
		float v00 = _dataterm_element_V00[k];
		float v01 = _dataterm_element_V01[k];
		float v10 = _dataterm_element_V10[k];
		float v11 = _dataterm_element_V11[k];
		float * ptr = _DataConstraints.ptr<float>(_DCc);       
		ptr[0] = _rowCount; ptr[1] = _x_index[(i-1)*_width+j-1]; ptr[2]= v00; _DCc = _DCc+1;
		ptr = _DataConstraints.ptr<float>(_DCc); 
		ptr[0] = _rowCount; ptr[1] = _x_index[(i-1)*_width+j]; ptr[2]= v01; _DCc = _DCc+1;
		ptr = _DataConstraints.ptr<float>(_DCc); 
		ptr[0] = _rowCount; ptr[1] = _x_index[i*_width+j-1]; ptr[2]= v10; _DCc = _DCc+1;
		ptr = _DataConstraints.ptr<float>(_DCc); 
		ptr[0] = _rowCount; ptr[1] = _x_index[i*_width+j]; ptr[2]= v11; _DCc = _DCc+1;		
		
		b.at<float>(_rowCount,0) = _dataterm_element_desPt[k].x;
		_rowCount = _rowCount+1;

		ptr = _DataConstraints.ptr<float>(_DCc);       
		ptr[0] = _rowCount; ptr[1] = _y_index[(i-1)*_width+j-1]; ptr[2]= v00; _DCc = _DCc+1;
		ptr = _DataConstraints.ptr<float>(_DCc); 
		ptr[0] = _rowCount; ptr[1] = _y_index[(i-1)*_width+j]; ptr[2]= v01; _DCc = _DCc+1;
		ptr = _DataConstraints.ptr<float>(_DCc); 
		ptr[0] = _rowCount; ptr[1] = _y_index[i*_width+j-1]; ptr[2]= v10; _DCc = _DCc+1;
		ptr = _DataConstraints.ptr<float>(_DCc); 
		ptr[0] = _rowCount; ptr[1] = _y_index[i*_width+j]; ptr[2]= v11; _DCc = _DCc+1;		
		
		b.at<float>(_rowCount,0) = _dataterm_element_desPt[k].y;
		_rowCount = _rowCount+1;        


	}
}

void ASAPWarping::Solve()
{
	cv::Mat b;
	CreateDataCons(b);
	int N = _SmoothConstraints.rows + _DataConstraints.rows;

	cv::Mat AMat(N,3,CV_32F);
	/*std::ofstream sc("smooth.txt");
	sc<<_SmoothConstraints;
	sc.close();
	std::ofstream dc("data.txt");
	dc<<_DataConstraints;
	dc.close();*/
	_SmoothConstraints.copyTo(AMat(cv::Rect(0,0,3,_SmoothConstraints.rows)));
	_DataConstraints.copyTo(AMat(cv::Rect(0,_SmoothConstraints.rows,3,_DataConstraints.rows)));

	std::vector<double> bm(b.rows),x;
	//std::ofstream bfile("b.txt");
	for(int i=0; i<bm.size(); i++)
	{
		bm[i] = b.at<float>(i,0);
		//bfile<<bm[i]<<std::endl;
	}
	/*bfile.close();*/
	//b.col(0).copyTo(bm);
	SolveSparse(AMat,bm,x);

	int hwidth = _columns/2;
	for(int i=0; i<_height; i++)
	{
		for(int j=0; j<_width; j++)
		{
			cv::Point2f pt(x[i*_width+j],x[hwidth+i*_width+j]);
			_destin->setVertex(i,j,pt);
		}
	}
}

void ASAPWarping::Warp(const cv::Mat& img1, cv::Mat& warpImg, int gap)
{
	_warpImg = cv::Mat::zeros(img1.rows+2*gap,img1.cols+2*gap,CV_8UC3);
	for(int i=1; i<_height; i++)
	{
		for(int j=1; j<_width; j++)
		{
			
			Quad qd1 = _source->getQuad(i,j);
			Quad qd2 = _destin->getQuad(i,j);
			quadWarp(img1,i-1,j-1,qd1,qd2);
		}
	}
	//warpImg = _warpImg.clone();
	cv::remap(img1,warpImg,_mapX,_mapY,CV_INTER_CUBIC);
}
void findHomographyDLT(std::vector<cv::Point2f>& f1, std::vector<cv::Point2f>& f2,cv::Mat& homography)
{
	homography.create(3,3,CV_64F);
	double* homoData = (double*)homography.data;
	cv::Mat dataM(2*f1.size(),9,CV_64F);
	double* ptr = (double*)dataM.data;
	int rowStep = dataM.step.p[0];
	for(int i=0; i<f1.size(); i++)
	{
		ptr[0] = ptr[1] = ptr[2] =0;
		ptr[3] = -1*f1[i].x;
		ptr[4] = -1*f1[i].y;
		ptr[5] = -1;
		ptr[6] = f2[i].y*f1[i].x;
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
	cv::Mat w,u,vt;
	//std::cout<<"A = "<<dataM<<std::endl;
	cv::SVDecomp(dataM,w,u,vt);
	//std::cout<<"vt = "<<vt<<std::endl;
	ptr = (double*)(vt.data + (vt.rows-1)*vt.step.p[0]);
	for(int i=0; i<9; i++)
		homoData[i] = ptr[i]/ptr[8];


}
void findHomographySVD(std::vector<cv::Point2f>& f1, std::vector<cv::Point2f>& f2,cv::Mat& homography)
{
	homography.create(3,3,CV_64F);
	double* homoData = (double*)homography.data;
	
	if (f1.size() != f2.size())
		return;
	int n = f1.size();
	if (n<4)
		return;
	cv::Mat h(9,2*n,CV_64F);
	cv::Mat rowsXY = cv::Mat::ones(3,n,CV_64F);
	double* ptr = rowsXY.ptr<double>(0);
	for(int i=0; i<n; i++)
		ptr[i] = -1*f1[i].x;
	ptr = rowsXY.ptr<double>(1);
	for(int i=0; i<n; i++)
		ptr[i] = -1*f1[i].y;
	ptr = rowsXY.ptr<double>(2);
	for(int i=0; i<n; i++)
		ptr[i] = -1;
	cv::Mat rows0 = cv::Mat::zeros(3,n,CV_64F);
	/*std::cout<<"rowsXY\n"<<rowsXY<<std::endl;*/
	rowsXY.copyTo(h(cv::Rect(0,0,rowsXY.cols,rowsXY.rows)));
	rows0.copyTo(h(cv::Rect(0,rowsXY.rows,n,3)));
	ptr = h.ptr<double>(6);
	for(int i=0; i<n; i++)
		ptr[i] = f1[i].x*f2[i].x;

	ptr = h.ptr<double>(7);
	for(int i=0; i<n; i++)
		ptr[i] = f1[i].y*f2[i].x;
	ptr = h.ptr<double>(8);
	for(int i=0; i<n; i++)
		ptr[i] = f2[i].x;

	rows0.copyTo(h(cv::Rect(n,0,n,3)));
	rowsXY.copyTo(h(cv::Rect(n,rows0.rows,rowsXY.cols,rowsXY.rows)));
	ptr = h.ptr<double>(6);
	for(int i=0; i<n; i++)
		ptr[n+i] = f1[i].x*f2[i].y;
	ptr = h.ptr<double>(7);
	for(int i=0; i<n; i++)
		ptr[n+i] = f1[i].y*f2[i].y;
	ptr = h.ptr<double>(8);
	for(int i=0; i<n; i++)
		ptr[i+n] = f2[i].y;
	/*std::cout<<"h\n"<<h<<std::endl;*/
	cv::Mat w,u,v;
	cv::SVD::compute(h,w,u,v,cv::SVD::FULL_UV);
	/*std::cout<<"w\n"<<w<<std::endl;
	std::cout<<"u\n"<<u<<std::endl;
	std::cout<<"v\n"<<v<<std::endl;*/
	cv::Mat ut;
	cv::transpose(u,ut);
	/*std::cout<<"ut\n"<<ut<<std::endl;*/
	ptr = (double*)(ut.data + (ut.rows-1)*ut.step.p[0]);
	for(int i=0; i<9; i++)
		homoData[i] = ptr[i]/ptr[8];
}
void ASAPWarping::quadWarp(const cv::Mat& img, int row, int col, Quad& q1, Quad& q2)
{
	float minx = q2.getMinX();
    float maxx = q2.getMaxX();
    float  miny = q2.getMinY();
    float  maxy = q2.getMaxY();
             
    std::vector<cv::Point2f> f1,f2;
	f1.push_back(q1.getV00());
	f1.push_back(q1.getV01());
	f1.push_back(q1.getV10());
	f1.push_back(q1.getV11());
             
	f2.push_back(q2.getV00());
	f2.push_back(q2.getV01());
	f2.push_back(q2.getV10());
	f2.push_back(q2.getV11());
	
	cv::Mat homography;	
	findHomographySVD(f1,f2,homography);
	cv::Mat invHomo = homography.inv();
	_homographies[row*(_width-1)+col] = homography.clone();
	_invHomographies[row*(_width-1)+col] =invHomo.clone();
	//std::cout<<homography;
	
             //qd = Quad(q2.V00,q2.V01,q2.V10,q2.V11);
           // _warpIm = myWarp(minx,maxx,miny,maxy,im,obj.warpIm,H,obj.gap);
            //_warpIm = uint8(obj.warpIm);
	/*if(gap > 0)
	{
		minx = floor(minx);
		miny =floor(miny);
	}
	else
	{
		minx = max(floor(minx),1);
		miny = max(floor(miny),1);
	}*/
	int width = img.cols;
	int height = img.rows;
	/*maxx = min(ceil(maxx),w);
	maxy = min(ceil(maxy),h);*/
	/*cv::Size size = cv::Size(maxx-minx,maxy-miny);
	cv::Mat map_x,map_y;*/
	
	double* ptr = (double*)invHomo.data;
	double* invPtr = (double*)homography.data;
	for(int i=0; i<_quadHeight; i++)
	{
		int r = row*_quadHeight+i;
		float* ptrX = _mapX.ptr<float>(r);
		float* ptrY = _mapY.ptr<float>(r);
		float* invPtrX = _invMapX.ptr<float>(r);
		float* invPtrY = _invMapY.ptr<float>(r);
		for(int j=0; j<_quadWidth; j++)
		{
			int c = col*_quadWidth+j;
			float x,y,w;
			x = c*ptr[0] + r*ptr[1] + ptr[2];
			y = c*ptr[3] + r*ptr[4] + ptr[5];
			w = c*ptr[6] + r*ptr[7] + ptr[8];
			x /=w;
			y/=w;
			ptrX[c] = x;
			ptrY[c] = y;

			x = c*invPtr[0] + r*invPtr[1] + invPtr[2];
			y = c*invPtr[3] + r*invPtr[4] + invPtr[5];
			w = c*invPtr[6] + r*invPtr[7] + invPtr[8];
			x /=w;
			y/=w;
			invPtrX[c] = x;
			invPtrY[c] = y;
			
		}
	}


}


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
		float theta = atan(dy/(dx+1e-6))/M_PI*180;
		if (theta<0)
			theta+=90;
		thetas[i] = theta;
		rads[i] = sqrt(dx*dx + dy*dy);
	
		max = rads[i] >max? rads[i] : max;
		min = rads[i]<min ? rads[i]: min;

	}
	float stepR = (max-min+1e-6)/DistSize;
	float stepT = 180/thetaSize;
	for(int i=0; i<f1.size(); i++)
	{
		int r = (int)((rads[i] - min)/stepR);
		int t = (int)(thetas[i]/stepT);
		r = r>DistSize? DistSize:r;
		t = t>thetaSize? thetaSize:t;
		int idx = t*DistSize+r;
		//std::cout<<idx<<std::endl;
		histogram[idx]++;
		ids[idx].push_back(i);
	
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
	cv::goodFeaturesToTrack(sGray,vf1,5000,0.05,2);
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