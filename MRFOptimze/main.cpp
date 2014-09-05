#include "PictureHandler.h"
#include "SLIC.h"
#include "GCoptimization.h"
#include "timer.h"
#include <math.h>
#include <hash_map>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/calib3d/calib3d.hpp>
typedef std::pair<int,int> Point2i;

bool compare(const Point2i& p1, const Point2i& p2)
{
	if (p1.first==p2.first)
		return p1.second<p2.second;
	else
		return p1.first < p2.first;
}

struct SuperPixel
{
	int idx;
	int lable;
	std::vector<Point2i> pixels;
	std::vector<SuperPixel*> neighbors;
	float avgColor;
	float ps;
};

typedef hash_map<int,SuperPixel*>SuperPixelMap;

const float theta(0.35);
const float lmd1(0.3);
const float lmd2(3.0);
cv::Mat g_prevImg;
cv::Mat g_currImg;

void findHomography(const cv::Mat& prevImg, const cv::Mat& currImg, cv::Mat & homography)
{
	std::vector<cv::Point2f> initial;   // initial position of tracked points
	std::vector<cv::Point2f> features;  // detected features
	int max_count(500);	  // maximum number of features to detect
	double qlevel(0.01);    // quality level for feature detection
	double minDist(10.);   // minimum distance between two feature points
	std::vector<uchar> status; // status of tracked features
	std::vector<float> err;    // error in tracking
	std::vector<cv::Point2f> points[2]; // tracked features from 0->1
	// detect the features
	cv::goodFeaturesToTrack(currImg, // the image 
		points[0],   // the output detected features
		max_count,  // the maximum number of features 
		qlevel,     // quality level
		minDist);   // min distance between two features

	// 2. track features
		cv::calcOpticalFlowPyrLK(currImg, prevImg, // 2 consecutive images
			points[0], // input point position in first image
			points[1], // output point postion in the second image
			status,    // tracking success
			err);      // tracking error

		// 2. loop over the tracked points to reject the undesirables
		int k=0;

		for( int i= 0; i < points[1].size(); i++ ) {

			// do we keep this point?
			if (status[i] == 1) {

				// keep this point in vector
				points[0][k] = points[0][i];
				points[1][k++] = points[1][i];
			}
		}
		// eliminate unsuccesful points
		points[0].resize(k);
		points[1].resize(k);

		//perspective transform
		std::vector<uchar> inliers(points[0].size(),0);
		homography= cv::findHomography(
			cv::Mat(points[0]), // corresponding
			cv::Mat(points[1]), // points
			inliers, // outputted inliers matches
			CV_RANSAC, // RANSAC method
			1.); // max distance to reprojection point
}


//save superpixels to an array with neighbor and pixels info
void GetSegment2DArray(SuperPixel *& superpixels, size_t & spSize, const int* lables, const int width,const int height)
{
	size_t size = width*height;
	superpixels = new SuperPixel[size];
	SuperPixelMap map;
	bool * visited = new bool[size];
	memset(visited,0,sizeof(bool)*size);
	//size of superpixels;
	spSize = 0;
	const int dx[8] = {-1, -1,  0,  1, 1, 1, 0, -1};
	const int dy[8] = { 0, -1, -1, -1, 0, 1, 1,  1};

	const size_t QSIZE = 1024*1024;
	Point2i *stack = new Point2i[QSIZE];
	int ptr = -1;
	stack[++ptr] = Point2i(0,0);
	while(ptr >= 0)
	{
		if (ptr>QSIZE)
			std::cout<<ptr<<std::endl;
		Point2i pt = stack[ptr--];
		int idx = pt.first + pt.second*width;
		if (!visited[idx])
		{
			visited[idx] = true;
			SuperPixel* SPPtr = NULL;
			if (map.find(lables[idx]) == map.end())
			{
				superpixels[spSize].lable = lables[idx];		
				superpixels[spSize].idx = spSize;
				SPPtr = &superpixels[spSize++];

				map[lables[idx]] = SPPtr;
			}
			else
				SPPtr = map[lables[idx]];	
			SPPtr->pixels.push_back(pt);

			for( int i=0; i<8; i++)
			{
				Point2i np = Point2i(pt.first+dx[i],pt.second+dy[i]);
				if (np.first<0)
					np.first = 0;
				if(np.first>=width)
					np.first = width-1;
				if(np.second<0)
					np.second = 0;
				if(np.second>=height)
					np.second = height-1;
				int nid = np.first + width*np.second;
				if (!visited[nid])
				{
					if (lables[nid] != lables[idx])	
					{
						//如果已经有此superpixel
						if (map.find(lables[nid]) != map.end())
						{
							/*map[lables[nid]]->pixels.push_back(np);*/
							if (find(map[lables[nid]]->neighbors.begin(),map[lables[nid]]->neighbors.end(),SPPtr)==map[lables[nid]]->neighbors.end())
								map[lables[nid]]->neighbors.push_back(SPPtr);
							if (find(SPPtr->neighbors.begin(),SPPtr->neighbors.end(),map[lables[nid]]) == SPPtr->neighbors.end())
								SPPtr->neighbors.push_back(map[lables[nid]]);
						}
						else
						{
							//添加到列表和图中
							superpixels[spSize].lable = lables[nid];				
							superpixels[spSize].idx = spSize;
							SuperPixel* Ptr = &superpixels[spSize++];

							map[lables[nid]] = Ptr;
							//建立联系
							SPPtr->neighbors.push_back(Ptr);
							Ptr->neighbors.push_back(SPPtr);
						}

					}
					stack[++ptr] = np;					
				}
			}
		}
	}
	delete[] visited;
	delete[] stack;
}

void ComputeAvgColor(SuperPixel* superpixels, size_t spSize, const int width, const int height,  const unsigned int* imgData, const unsigned char* maskData)
{
	cv::Mat psMat(height,width,CV_8U);
	psMat = cv::Scalar(0);
	uchar* psImg = psMat.data;
	cv::Mat avgMat(height,width,CV_8U);
	avgMat = cv::Scalar(0);
	uchar* avgImg = avgMat.data;
	unsigned char* cimgData = (unsigned char*)imgData;
	for(int i=0; i<spSize; i++)
	{
		float tmp = 0;
		float mtmp = 0;
		for( int j=0; j<superpixels[i].pixels.size(); j++)
		{
			int idx = (superpixels[i].pixels[j].first + superpixels[i].pixels[j].second*width);
			int idxC = (superpixels[i].pixels[j].first + superpixels[i].pixels[j].second*width)*4;
			for (int c=1; c<4; c++)
			{
				/*std::cout<<(int)cimgData[idxC+c]<<std::endl; */
				tmp += cimgData[idxC+c];
			}
			if (maskData[idx] == 0xff)
				mtmp++;
		}
		superpixels[i].avgColor = tmp/superpixels[i].pixels.size()/3;
		superpixels[i].ps = mtmp/superpixels[i].pixels.size();
		for(int p=0; p<superpixels[i].pixels.size(); p++)
		{
			int idx = superpixels[i].pixels[p].first + superpixels[i].pixels[p].second*width;
			psImg[idx] = uchar(superpixels[i].ps*255);
			avgImg[idx] = uchar(superpixels[i].avgColor);
		}
	}
	//cv::imwrite("prob.jpg",psMat);
	//cv::imwrite("avg.jpg",avgMat);
}
void MaxFlowOptimize(SuperPixel* spPtr, int num_pixels,float beta, int num_labels,const int width, const int height,int *result)
{
	size_t num_edges = 0;
	for(int i=0; i<num_pixels; i++)
		num_edges += spPtr[i].neighbors.size();
	num_edges /= 2;

	typedef Graph<float,float,float> GraphType;
	GraphType *g = new GraphType(/*estimated # of nodes*/ num_pixels, /*estimated # of edges*/ num_edges); 

	for(int i=0; i<num_pixels; i++)
	{
		g->add_node();
		float d = min(1.0f,theta*spPtr[i].ps*2);
		d = max(1e-20f,d);
		float d1 = -log(d);
		float d2 =  - log(1-d);
		g->add_tweights(i,d1,d2);
	}
	for(int i=0; i<num_pixels; i++)
	{
		for(int j=0; j<spPtr[i].neighbors.size(); j++)
		{
			if (i < spPtr[i].neighbors[j]->idx)
			{
				float energy = (lmd1+lmd2*exp(-beta*abs(spPtr[i].avgColor-spPtr[i].neighbors[j]->avgColor)));
				g->add_edge(i,spPtr[i].neighbors[j]->idx,energy,energy);
			}
		}
	}

	float flow = g -> maxflow();
	for ( int  i = 0; i < num_pixels; i++ )
		result[i] = g->what_segment(i) == GraphType::SINK ? 0x1 : 0;
}
void GraphCutOptimize(SuperPixel* spPtr, int num_pixels,float beta, int num_labels,const int width, const int height,int *result)
{
	// first set up the array for data costs
	float* data = new float[num_pixels*num_labels];

	for(int i=0; i<num_pixels; i++)
	{
		for(int j=0; j<num_labels; j++)
		{
			float d = min(1.0f,theta*spPtr[i].ps*2);
			d = max(1e-20f,d);
			float d1 = -log(d)*j;
			float d2 =  - log(1-d)*(1-j);
			data[i*num_labels + j] =(d1+d2);
		}
	}
	// next set up the array for smooth costs
	float *smooth = new float[num_labels*num_labels];
	for ( int l1 = 0; l1 < num_labels; l1++ )
		for (int l2 = 0; l2 < num_labels; l2++ )
			smooth[l1+l2*num_labels] = abs(l1-l2);


	try{
		GCoptimizationGeneralGraph *gc = new GCoptimizationGeneralGraph(num_pixels,num_labels);
		gc->setDataCost(data);
		//gc->setSmoothCost(&smoothFn);
		gc->setSmoothCost(smooth);
		//std::ofstream file("out.txt");
		// now set up a grid neighborhood system
		for(int i=0; i<num_pixels; i++)
		{
			for(int j=0; j<spPtr[i].neighbors.size(); j++)
			{				
				if (i>spPtr[i].neighbors[j]->idx)
				{
					float energy = (lmd1+lmd2*exp(-beta*abs(spPtr[i].avgColor-spPtr[i].neighbors[j]->avgColor)));
					//file<<energy<<std::endl;
					gc->setNeighbors(i,spPtr[i].neighbors[j]->idx,energy);
				}
			}
		}
		//file.close();
		//printf("\nBefore optimization energy is %d",gc->compute_energy());
		//printf("\nBefore optimization  data energy is %d",gc->giveDataEnergy());
		//printf("\nBefore optimization smooth energy is %d",gc->giveSmoothEnergy());
		gc->expansion(5);// run expansion for 2 iterations. For swap use gc->swap(num_iterations);
		//printf("\nAfter optimization energy is %d",gc->compute_energy());
		//printf("\nAfter optimization  data energy is %d",gc->giveDataEnergy());
		//printf("\nAfter optimization smooth energy is %d",gc->giveSmoothEnergy());

		for ( int  i = 0; i < num_pixels; i++ )
			result[i] = gc->whatLabel(i);
		delete gc;
	}
	catch (GCException e){
		e.Report();
	}


	delete [] smooth;
	delete [] data;


}

void TestSuperpixel(string& filename, string& savename,  string& saveLocation)
{
	int width(0);
	int height(0);
	UINT* img = NULL;
	PictureHandler	picHand;
	picHand.GetPictureBuffer( filename, img, width, height );
	cv::Mat Img = cv::imread(filename);
	uchar* camData = new uchar[Img.total()*4];
	cv::Mat continuousRGBA(Img.size(), CV_8UC4, camData);	
	cv::cvtColor(Img,continuousRGBA,CV_BGR2BGRA,4);
	unsigned int* data = (unsigned*)camData;


	size_t sz = width*height;
	int* labels = new int[sz];
	int numlabels(0);
	SLIC slic;
	slic.PerformSLICO_ForGivenK(data, width, height, labels, numlabels, 2000, 20);//for a given number K of superpixels
	//slic.PerformSLICO_ForGivenStepSize(img, width, height, labels, numlabels, m_stepsize, m_compactness);//for a given grid step size
	//slic.DrawContoursAroundSegments(img, labels, width, height, 0);//for black contours around superpixels
	slic.DrawContoursAroundSegmentsTwoColors(data, labels, width, height);//for black-and-white contours around superpixels
	slic.SaveSuperpixelLabels(labels,width,height,savename+".dat",saveLocation);

	picHand.SavePicture(data, width, height, savename, saveLocation, 1, "_SLICO");// 0 is for BMP and 1 for JPEG)
	SuperPixel* spPtr = NULL;
	size_t spSize(0);
	GetSegment2DArray(spPtr,spSize,labels,width,height);
	std::cout<<std::endl;
	std::cout<<spSize<<std::endl;
	std::cout<<spPtr[0].lable<<std::endl;
	std::cout<<spPtr[0].pixels.size()<<std::endl;
	std::cout<<spPtr[0].neighbors.size()<<std::endl;

	cv::Mat neighborImg(height,width,CV_8U);
	int c = 250;
	bool suc = true;
	for(int k=0; k<spSize; k++)
	{
		if (k==c)
		{
			for(int i = 0; i<spPtr[k].pixels.size(); i++)
			{
				neighborImg.data[spPtr[k].pixels[i].first + width*spPtr[k].pixels[i].second] = 0xff;
			}
			for( int i=0; i<spPtr[k].neighbors.size(); i++)
			{
				for(int j=0; j<spPtr[k].neighbors[i]->pixels.size(); j++)
					neighborImg.data[spPtr[k].neighbors[i]->pixels[j].first + width*spPtr[k].neighbors[i]->pixels[j].second] = 0xcc;
				/*for(int j=0; j<spPtr[spPtr[k].neighbors[i]->idx].pixels.size(); j++)
					neighborImg.data[spPtr[spPtr[k].neighbors[i]->idx].pixels[j].first + width*spPtr[spPtr[k].neighbors[i]->idx].pixels[j].second] = 0x55;*/
				
			}
			cv::imwrite("neighborImg.jpg",neighborImg);
		
		}
		std::sort(spPtr[k].pixels.begin(),spPtr[k].pixels.end(),compare);
		std::vector<Point2i> pixel0;
		for(int i=0; i<width; i++)
		{
			for(int j=0; j<height; j++)
			{
				int idx = i + j*width;
				if (labels[idx] == spPtr[k].lable)
					pixel0.push_back(Point2i(i,j));
			}
		}

		for(int i=0; i<pixel0.size(); i++)
		{
			if(pixel0[i].first!=spPtr[k].pixels[i].first || pixel0[i].second!= spPtr[k].pixels[i].second)
			{
				suc = false;
				break;
			}
		}
		if (!suc)
			break;
	}
	if (suc)
		std::cout<<"yes"<<endl;

	if(labels) delete [] labels;
	if(camData) delete [] camData;
}
////////////////////////////////////////////////////////////////////////////////
// in this version, set data and smoothness terms using arrays
// grid neighborhood is set up "manually". Uses spatially varying terms. Namely
// V(p1,p2,l1,l2) = w_{p1,p2}*[min((l1-l2)*(l1-l2),4)], with 
// w_{p1,p2} = p1+p2 if |p1-p2| == 1 and w_{p1,p2} = p1*p2 if |p1-p2| is not 1

void GeneralGraph_DArraySArraySpatVarying(int width,int height,int num_pixels,int num_labels)
{
	int *result = new int[num_pixels];   // stores result of optimization

	// first set up the array for data costs
	float *data = new float[num_pixels*num_labels];
	for ( int i = 0; i < num_pixels; i++ )
		for (int l = 0; l < num_labels; l++ )
			if (i < 25 ){
				if(  l == 0 ) data[i*num_labels+l] = 0;
				else data[i*num_labels+l] = 10;
			}
			else {
				if(  l == 5 ) data[i*num_labels+l] = 0;
				else data[i*num_labels+l] = 10;
			}
			// next set up the array for smooth costs
			float *smooth = new float[num_labels*num_labels];
			for ( int l1 = 0; l1 < num_labels; l1++ )
				for (int l2 = 0; l2 < num_labels; l2++ )
					smooth[l1+l2*num_labels] = (l1-l2)*(l1-l2) <= 4  ? (l1-l2)*(l1-l2):4;


			try{
				GCoptimizationGeneralGraph *gc = new GCoptimizationGeneralGraph(num_pixels,num_labels);
				gc->setDataCost(data);
				gc->setSmoothCost(smooth);

				// now set up a grid neighborhood system
				// first set up horizontal neighbors
				for (int y = 0; y < height; y++ )
					for (int  x = 1; x < width; x++ ){
						int p1 = x-1+y*width;
						int p2 =x+y*width;
						gc->setNeighbors(p1,p2,p1+p2);
					}

					// next set up vertical neighbors
					for (int y = 1; y < height; y++ )
						for (int  x = 0; x < width; x++ ){
							int p1 = x+(y-1)*width;
							int p2 =x+y*width;
							gc->setNeighbors(p1,p2,p1*p2);
						}

						printf("\nBefore optimization energy is %d",gc->compute_energy());
						printf("\nBefore optimization  data energy is %d",gc->giveDataEnergy());
						printf("\nBefore optimization smooth energy is %d",gc->giveSmoothEnergy());
						gc->expansion(20);// run expansion for 2 iterations. For swap use gc->swap(num_iterations);
						printf("\nAfter optimization energy is %d",gc->compute_energy());
						printf("\nAfter optimization  data energy is %d",gc->giveDataEnergy());
						printf("\nAfter optimization smooth energy is %d",gc->giveSmoothEnergy());

						for ( int  i = 0; i < num_pixels; i++ )
							result[i] = gc->whatLabel(i);

						delete gc;
			}
			catch (GCException e){
				e.Report();
			}

			delete [] result;
			delete [] smooth;
			delete [] data;


}
void testGCO()
{
	int width = 10;
	int height = 5;
	int num_pixels = width*height;
	int num_labels = 7;
	//Will pretend our graph is general, and set up a neighborhood system
	// which actually is a grid. Also uses spatially varying terms
	GeneralGraph_DArraySArraySpatVarying(width,height,num_pixels,num_labels);
}
void testSuperpixel()
{
	const char filename[] = "..\\PTZ\\input0\\in000483.jpg";
	string savename = "output";
	string saveLocation = ".\\";
	TestSuperpixel(string(filename),savename,saveLocation);
}
void MaskHomographyTest(cv::Mat& mCurr, cv::Mat& curr, cv::Mat & prev, cv::Mat& homography)
{
	float threshold = 5.0;
	std::vector<cv::Point2f> currPoints, trackedPoints;
	std::vector<uchar> status; // status of tracked features
	std::vector<float> err;    // error in tracking
	for(int i=0; i<mCurr.cols; i++)
	{
		for(int j=0; j<mCurr.rows; j++)
			if(mCurr.data[i + j*mCurr.cols] == 0xff)
				currPoints.push_back(cv::Point2f(i,j));
	}
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

		float distance = 0;
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
			distance += d;
			if (d < threshold)
			{
				mCurr.data[(int)currPoints[i].x+(int)currPoints[i].y*mCurr.cols] = 0x0f;
				
			}

		}
		distance /= k;
		char name[20];
		cv::imshow("win",mCurr);
		cv::waitKey();
}
//用MRF对前景结果进行优化
void MRFOptimize(const string& originalImgName, const string& maskImgName, const string& resultImgName)
{
	//superpixel
	cv::Mat Img = cv::imread(originalImgName);
	
	uchar* camData = new uchar[Img.total()*4];
	cv::Mat continuousRGBA(Img.size(), CV_8UC4, camData);	
	cv::cvtColor(Img,continuousRGBA,CV_BGR2BGRA,4);
	
	//convert to ARGB data array
	unsigned int* idata = new unsigned int[Img.total()];
	for(int i=0; i<Img.cols; i++)
	{
		unsigned char tmp[4];
		
		for(int j=0; j<Img.rows; j++)
		{
			for(int k=0; k<4; k++)
				tmp[k] = continuousRGBA.data[continuousRGBA.step[0]*j + continuousRGBA.step[1]*i + continuousRGBA.elemSize1()*k];
			idata[i + j*Img.cols] = tmp[3]<<24 | tmp[2]<<16| tmp[1]<<8 | tmp[0];
		}
	}
		
	cv::Mat maskImg = cv::imread(maskImgName);
	cv::cvtColor(maskImg,maskImg,CV_BGR2GRAY);
	maskImg.convertTo(maskImg,CV_8U);
	const unsigned char* maskImgData = maskImg.data;
	int width = Img.cols;
	int height = Img.rows;

	size_t sz = width*height;
	int* labels = new int[sz];
	int numlabels(0);
#ifdef REPORT
	nih::Timer timer;
	timer.start();
#endif
	SLIC slic;
	slic.PerformSLICO_ForGivenK(idata, width, height, labels, numlabels, 8000, 20);//for a given number K of superpixels
#ifdef REPORT
	timer.stop();
	std::cout<<"SLIC SuperPixel "<<timer.seconds()<<std::endl;
#endif
	//slic.DrawContoursAroundSegmentsTwoColors(idata, labels, width, height);//for black-and-white contours around superpixels
	//slic.SaveSuperpixelLabels(labels,width,height,savename+".dat",saveLocation);	
#ifdef REPORT
	
	timer.start();
#endif
	SuperPixel* spPtr = NULL;
	size_t spSize(0);
	GetSegment2DArray(spPtr,spSize,labels,width,height);
	delete[] labels;
#ifdef REPORT
	timer.stop();
	std::cout<<"GetSegment2DArray  "<<timer.seconds()<<std::endl;
#endif

#ifdef REPORT
	
	timer.start();
#endif
	ComputeAvgColor(spPtr,spSize,width,height,idata,maskImgData);
	float avgE = 0;
	size_t count = 0;
	for(int i=0; i<spSize; i++)
	{
		for (int j=0; j< spPtr[i].neighbors.size(); j++)
		{
			avgE += abs(spPtr[i].avgColor-spPtr[i].neighbors[j]->avgColor);
			count++;

		}
	}
	avgE /= count;
	avgE = 1/(2*avgE);

#ifdef REPORT
	timer.stop();
	std::cout<<"ComputeAvgColor  "<<timer.seconds()<<std::endl;
#endif

#ifdef REPORT
	
	timer.start();
#endif
	int *result = new int[spSize];   // stores result of optimization
	GraphCutOptimize(spPtr,spSize,avgE,2,width,height,result);
	//MaxFlowOptimize(spPtr,spSize,avgE,2,width,height,result);
#ifdef REPORT
	timer.stop();
	std::cout<<"GraphCutOptimize  "<<timer.seconds()<<std::endl;
#endif

#ifdef REPORT
	
	timer.start();
#endif
	cv::Mat img(height,width,CV_8U);
	img = cv::Scalar(0);
	unsigned char* imgPtr = img.data;
	for(int i=0; i<spSize; i++)
	{
		if(result[i] == 1)			
		{
			for(int j=0; j<spPtr[i].pixels.size(); j++)
			{
				int idx = spPtr[i].pixels[j].first + spPtr[i].pixels[j].second*width;
				imgPtr[idx] = 0xff;
			}
		}	
	}
	cv::imwrite(resultImgName,img);
#ifdef REPORT
	timer.stop();
	std::cout<<"imwrite  "<<timer.seconds()<<std::endl;
#endif
	delete[] result;
	delete[] camData;
	delete[] spPtr;
	delete[] idata;
}

void HomoTest(const char* originalImgName, const char* maskImgName)
{
	cv::Mat currImg = cv::imread(originalImgName);
	cv::cvtColor(currImg,g_currImg,CV_BGR2GRAY);

	cv::Mat currFImg = cv::imread(maskImgName);
	cv::cvtColor(currFImg,currFImg,CV_BGR2GRAY);
	if (g_prevImg.empty())
		g_currImg.copyTo(g_prevImg);
	cv::Mat homography;
	findHomography(g_prevImg,g_currImg,homography);
	MaskHomographyTest(currFImg,g_currImg,g_prevImg,homography);

	cv::swap(g_prevImg,g_currImg);

}
void testHomo()
{
	using namespace std;
	char imgFileName[150];
	char maskFileName[150];
	char resultFileName[150];
	for(int i=188; i<=200;i++)
	{
		sprintf(imgFileName,"..\\ptz\\input3\\in%06d.jpg",i);
		sprintf(maskFileName,"..\\result\\subsensem\\input3\\bin%06d.png",i);
		//sprintf(maskFileName,"H:\\changeDetection2014\\dataset2014\\dataset\\PTZ\\continuousPan\\groundtruth\\gt%06d.png",i);
		HomoTest(imgFileName,maskFileName);
	}
}
int main()
{
	/*testGCO();*/
	/*testSuperpixel();*/
	testHomo();
	return 0;
	using namespace std;
	char imgFileName[150];
	char maskFileName[150];
	char resultFileName[150];
	for(int i=470; i<=1700;i++)
	{
		sprintf(imgFileName,"..\\baseline\\input0\\in%06d.jpg",i);
		sprintf(maskFileName,"H:\\changeDetection2012\\SOBS_20\\results\\baseline\\highway\\bin%06d.png",i);
		sprintf(resultFileName,"..\\result\\SubsenseMMRF\\baseline\\input0\\bin%06d.png",i);
		MRFOptimize(string(imgFileName),string(maskFileName),string(resultFileName));
	}





	return 0;
}