#include "Test.h"
#include "flowIO.h"
#include "ASAPWarping.h"
void testCudaGpu()
{
	try

	{

		cv::Mat src_host = cv::imread("in000001.jpg");

		cv::gpu::GpuMat dst, src;

		src.upload(src_host);

		cv::gpu::cvtColor(src,src,CV_BGR2GRAY);

		cv::gpu::threshold(src, dst, 128.0, 255.0, CV_THRESH_BINARY);
		
		//cv::Mat result_host = dst;

		cv::Mat result_host;

		dst.download(result_host);

		cv::imshow("Result", result_host);

		cv::waitKey();

	}

	catch(const cv::Exception& ex)

	{

		std::cout << "Error: " << ex.what() << std::endl;

	}
}

void CpuSuperpixel(unsigned int* data, int width, int height, int step, float alpha = 0.9)
{
	int size = width*height;
	int* labels = new int[size];
	unsigned int* idata = new unsigned int[size];
	memcpy(idata,data,sizeof(unsigned int)*size);
	int numlabels(0);
	ComSuperpixel CS;
	//CS.Superpixel(idata,width,height,7000,0.9,labels);
#ifdef REPORT
	nih::Timer timer;
	timer.start();
#endif
	CS.SuperpixelLattice(idata,width,height,step,alpha,numlabels,labels);
#ifdef REPORT
	timer.stop();
	std::cout<<"SLIC SuperPixel "<<timer.seconds()<<std::endl;
#endif
	SLIC aslic;
	aslic.DrawContoursAroundSegments(idata, labels, width, height,0x00ff00);
	PictureHandler handler;
	handler.SavePicture(idata,width,height,std::string("cpusuper.jpg"),std::string(".\\"));
	aslic.SaveSuperpixelLabels(labels,width,height,std::string("cpuSp.txt"),std::string(".\\"));
	delete[] labels;
	delete[] idata;
}

void TestSuperpixel()
{
	using namespace cv;
	Mat img = imread("..//ptz//input0//in000090.jpg");
	cv::cvtColor(img,img,CV_BGR2BGRA);
	//cv::resize(img,img,cv::Size(16,16));
	uchar4* imgData = new uchar4[img.rows*img.cols];
	unsigned int* idata = new unsigned int[img.rows*img.cols];
	unsigned char tmp[4];
	for(int i=0; i< img.cols; i++)
	{
		
		for(int j=0; j<img.rows; j++)
		{
			int idx = img.step[0]*j + img.step[1]*i;
			for(int k=0; k<4; k++)
				tmp[k] = img.data[idx + img.elemSize1()*k];
			imgData[i + j*img.cols].x = tmp[0];
			imgData[i + j*img.cols].y = tmp[1];
			imgData[i + j*img.cols].z = tmp[2];
			imgData[i + j*img.cols].w = tmp[3];
			
			
			idata[i + j*img.cols] = tmp[3]<<24 | tmp[2]<<16| tmp[1]<<8 | tmp[0];
		}
	}
	CpuSuperpixel(idata,img.cols,img.rows,15);
	GpuSuperpixel gs(img.cols,img.rows,15);
	int num(0);
	int* labels = new int[img.rows*img.cols];

	GpuTimer timer;
	SLIC aslic;	
	PictureHandler handler;

	timer.Start();
	gs.Superpixel(imgData,num,labels);
	timer.Stop();
	std::cout<<timer.Elapsed()<<"ms"<<std::endl;
	
	aslic.DrawContoursAroundSegments(idata, labels, img.cols,img.rows,0x00ff00);
	handler.SavePicture(idata,img.cols,img.rows,std::string("GpuSp.jpg"),std::string(".\\"));
	aslic.SaveSuperpixelLabels(labels,img.cols,img.rows,std::string("GpuSp.txt"),std::string(".\\"));
	
	memset(labels,0,sizeof(int)*img.rows*img.cols);
	timer.Start();
	gs.SuperpixelLattice(imgData,num,labels);
	timer.Stop();
	std::cout<<timer.Elapsed()<<"ms"<<std::endl;
	aslic.DrawContoursAroundSegments(idata, labels, img.cols,img.rows,0x00ff00);
	handler.SavePicture(idata,img.cols,img.rows,std::string("GpuSpLatttice.jpg"),std::string(".\\"));
	aslic.SaveSuperpixelLabels(labels,img.cols,img.rows,std::string("GpuSpLatttice.txt"),std::string(".\\"));
	delete[] labels;
	delete[] idata;
	delete[] imgData;
}

void MRFOptimization()
{
	using namespace std;
	char imgFileName[150];
	char maskFileName[150];
	char featureMaskFileName[150];
	char resultFileName[150];
	int cols = 320;
	int rows = 240;
	GpuSuperpixel gs(cols,rows,5);
	MRFOptimize optimizer(cols,rows,5);
	nih::Timer timer;
	timer.start();
	int start = 1;
	int end = 1130;
	for(int i=start; i<=end;i++)
	{
		sprintf(imgFileName,"..\\ptz\\input0\\in%06d.jpg",i);
		sprintf(maskFileName,"..\\result\\subsensex\\ptz\\input0\\o\\bin%06d.png",i);
		sprintf(featureMaskFileName,"..\\result\\subsensex\\ptz\\input0\\features\\features%06d.jpg",i);
		sprintf(resultFileName,"..\\result\\SubsenseMMRF\\ptz\\input0\\bin%06d.png",i);
		
		/*sprintf(imgFileName,"..\\baseline\\input0\\in%06d.jpg",i);
		sprintf(maskFileName,"..\\result\\sobs\\baseline\\input0\\bin%06d.png",i);
		sprintf(resultFileName,"..\\result\\SubsenseMMRF\\baseline\\input0\\bin%06d.png",i);*/
		//optimizer.Optimize(&gs,string(imgFileName),string(maskFileName),string(resultFileName));
		optimizer.Optimize(&gs,string(imgFileName),string(maskFileName),string(featureMaskFileName),string(resultFileName));
	}
	timer.stop();
	std::cout<<(end-start+1)/timer.seconds()<<" fps\n";
}
void warmUpDevice()
{
	cv::Mat cpuMat = cv::Mat::ones(100,100,CV_8UC3);
	cv::gpu::GpuMat gmat;
	gmat.upload(cpuMat);
	gmat.download(cpuMat);
}
void TestRandom()
{
	int width(8);
	int height(6);
	int* d_rand,*h_rand;
	cudaMalloc(&d_rand,width*height*2*sizeof(int));
	h_rand = new int[width*height*2];
	TestRandNeighbour(width,height,d_rand);	
	cudaMemcpy(h_rand,d_rand,sizeof(int)*width*height*2,cudaMemcpyDeviceToHost);
	for(int i=0; i<width*height; i++)
	{
		std::cout<<h_rand[2*i]<<","<<h_rand[2*i+1]<<" ";
		if ((2*i+1)%width == 0)
			std::cout<<std::endl;
	}
	std::cout<<"---------\n";
	TestRandNeighbour(width,height,d_rand);	
	cudaMemcpy(h_rand,d_rand,sizeof(int)*width*height*2,cudaMemcpyDeviceToHost);
	for(int i=0; i<width*height; i++)
	{
		std::cout<<h_rand[2*i]<<","<<h_rand[2*i+1]<<" ";
		if ((2*i+1)%width == 0)
			std::cout<<std::endl;
	}
	std::cout<<"---------\n";
	int x_rand,y_rand;
	cv::Size size(width,height);
	for(int i=0; i<width; i++)
	{
		for(int j=0; j<height; j++)
		{
			getRandNeighborPosition_3x3(x_rand,y_rand,i,j,2,size);
			std::cout<<x_rand<<","<<y_rand<<" ";

		}
		std::cout<<"\n";
	}
	std::cout<<"---------\n";
	for(int i=0; i<width; i++)
	{
		for(int j=0; j<height; j++)
		{
			getRandNeighborPosition_3x3(x_rand,y_rand,i,j,2,size);
			std::cout<<x_rand<<","<<y_rand<<" ";

		}
		std::cout<<"\n";
	}
	cudaFree(d_rand);
	delete[] h_rand;
}
void TestGpuSubsense()
{
	
	warmUpDevice();
	VideoProcessor processor;
	
	// Create feature tracker instance
	SubSenseBSProcessor tracker;
	std::vector<std::string> fileNames;
	int start = 180;
	int end = 1130;
	for(int i=start; i<=end;i++)
	{
		char name[50];
		sprintf(name,"..\\ptz\\input3\\in%06d.jpg",i);
		//sprintf(name,"..\\PTZ\\input4\\drive1_%03d.png",i);
		fileNames.push_back(name);
	}
	// Open video file
	processor.setInput(fileNames);
	//processor.setInput("..\\ptz\\woman.avi");
	// set frame processor
	processor.setFrameProcessor(&tracker);

	processor.dontDisplay();
	// Declare a window to display the video
	//processor.displayOutput("Tracked Features");

	// Play the video at the original frame rate
	//processor.setDelay(1000./processor.getFrameRate());
	processor.setDelay(0);

	nih::Timer timer;
	timer.start();
	// Start the process
	processor.run();
	timer.stop();

	std::cout<<(end-start+1)/timer.seconds()<<" fps"<<std::endl;
	

	cv::waitKey();


	

}
void TestMotionEstimate()
{
	char fileName[100];
	int start = 2;
	int end = 40;
	cv::Mat curImg,prevImg,transM;
	MotionEstimate me(640,480,5);
	cv::Mat mask;
	for(int i=start; i<=end; i++)
	{
		sprintf(fileName,"..//moseg//people1//in%06d.jpg",i);
		curImg = cv::imread(fileName);
		sprintf(fileName,"..//moseg//people1//in%06d.jpg",i-1);
		prevImg = cv::imread(fileName);
		//me.EstimateMotionMeanShift(curImg,prevImg,transM,mask);
		me.EstimateMotion(curImg,prevImg,transM,mask);
		//me.EstimateMotionHistogram(curImg,prevImg,transM,mask);
		sprintf(fileName,".//features//people1//features%06d.jpg",i);
		cv::imwrite(fileName,mask);
		/*cv::imshow("curImg",curImg);
		cv::imshow("prevImg",prevImg);*/
		cv::waitKey();
	}
}
void TestRegionGrowing()
{
	std::vector<cv::Point2f> seeds;
	seeds.push_back(cv::Point2f(30,66));
	cv::Mat img = cv::imread("..//ptz//input0in000225.jpg");
	cv::Mat result(img.size(),CV_8U);
	result = cv::Scalar(0);
	RegionGrowing(seeds,img,result);
	cv::imshow("region grow",result);
	cv::waitKey();
}
template <typename T> inline T clamp (T x, T a, T b)
{
    return ((x) > (a) ? ((x) < (b) ? (x) : (b)) : (a));
}

template <typename T> inline T mapValue(T x, T a, T b, T c, T d)
{
    x = clamp(x, a, b);
    return c + (d - c) * (x - a) / (b - a);
}
static void getFlowField(const Mat& u, const Mat& v, Mat& flowField)
{
    float maxDisplacement = 1.0f;

    for (int i = 0; i < u.rows; ++i)
    {
        const float* ptr_u = u.ptr<float>(i);
        const float* ptr_v = v.ptr<float>(i);

        for (int j = 0; j < u.cols; ++j)
        {
            float d = max(fabsf(ptr_u[j]), fabsf(ptr_v[j]));

            if (d > maxDisplacement)
                maxDisplacement = d;
        }
    }

    flowField.create(u.size(), CV_8UC4);

    for (int i = 0; i < flowField.rows; ++i)
    {
        const float* ptr_u = u.ptr<float>(i);
        const float* ptr_v = v.ptr<float>(i);


        Vec4b* row = flowField.ptr<Vec4b>(i);

        for (int j = 0; j < flowField.cols; ++j)
        {
            row[j][0] = 0;
            row[j][1] = static_cast<unsigned char> (mapValue (-ptr_v[j], -maxDisplacement, maxDisplacement, 0.0f, 255.0f));
            row[j][2] = static_cast<unsigned char> (mapValue ( ptr_u[j], -maxDisplacement, maxDisplacement, 0.0f, 255.0f));
            row[j][3] = 255;
        }
    }
}
void GpuDenseOptialFlow::DenseOpticalFlow(const cv::Mat& curImg, const cv::Mat& prevImg, cv::Mat& flow)
{
	cv::Mat gray1, gray2;
	if (curImg.channels() == 3)
		cv::cvtColor(curImg,gray1,CV_BGR2GRAY);
	else
		gray1 = curImg.clone();
	if (prevImg.channels() ==3)
		cv::cvtColor(prevImg,gray2,CV_BGR2GRAY);
	else
		gray2 = prevImg.clone();
	cv::gpu::GpuMat dCurImg(gray1);
	cv::gpu::GpuMat dPrevImg(gray2);
	
	cv::gpu::GpuMat du,dv;
	cv::gpu::PyrLKOpticalFlow dPyrLK;

	dPyrLK.dense(dCurImg,dPrevImg,du,dv);
	
	std::vector<cv::Mat> flows(2);
	du.download(flows[0]);
	dv.download(flows[1]);
	/*cv::Mat flowField;
	getFlowField(flows[0],flows[1],flowField);
	cv::imwrite("flow.jpg",flowField);
	*/
	cv::merge(flows,flow);
}
void SFDenseOptialFlow::DenseOpticalFlow(const cv::Mat& curImg, const cv::Mat& prevImg, cv::Mat& flow)
{
	cv::Mat img0,img1;
	cv::cvtColor(curImg,img0,CV_BGR2GRAY);
	cv::cvtColor(prevImg,img1,CV_BGR2GRAY);
	cv::calcOpticalFlowSF(img0,img1,flow,3,2,4);
	cv::Mat flowField;
	cv::Mat flows[2];
	cv::split(flow,flows);
	getFlowField(flows[0],flows[1],flowField);
	cv::imwrite("flow_SF.jpg",flowField);
}
void FarnebackDenseOptialFlow::DenseOpticalFlow(const cv::Mat& curImg, const cv::Mat& prevImg, cv::Mat& flow)
{
	cv::Mat img0,img1;
	if (curImg.channels() == 3)
	{
		cv::cvtColor(curImg,img0,CV_BGR2GRAY);
		cv::cvtColor(prevImg,img1,CV_BGR2GRAY);
	}
	else
	{
		img0 = curImg;
		img1 = prevImg;
	}
	//cv::calcOpticalFlowSF(img0,img1,flow,3,2,55);
	cv::calcOpticalFlowFarneback(img0,img1,flow,0.5, 5, 5, 5, 5, 1.2, 0);
	/*cv::Mat flowField;
	cv::Mat flows[2];
	cv::split(flow,flows);
	getFlowField(flows[0],flows[1],flowField);
	cv::imwrite("flow_Farnback.jpg",flowField);*/
}
void EPPMDenseOptialFlow::DenseOpticalFlow(const cv::Mat& curImg, const cv::Mat& prevImg, cv::Mat& flow)
{
	cv::Mat img0,img1;
	if (curImg.channels() == 3)
	{
		cv::cvtColor(curImg,img0,CV_BGR2GRAY);
		cv::cvtColor(prevImg,img1,CV_BGR2GRAY);
	}
	else
	{
		img0 = curImg;
		img1 = prevImg;
	}
	cv::imwrite("img0.png",img0);
	cv::imwrite("img1.png",img1);
	std::system(" EPPM_flow.exe  img0.png img1.png flow.flo flow.png ");
	ReadFlowFile(flow,"flow.flo");
}
void GetHomography(const cv::Mat& gray,const cv::Mat& pre_gray, cv::Mat& homography)
{
	int max_count = 50000;	  // maximum number of features to detect
	double qlevel = 0.05;    // quality level for feature detection
	double minDist = 2;   // minimum distance between two feature points
	std::vector<uchar> status; // status of tracked features
	std::vector<float> err;    // error in tracking
	std::vector<cv::Point2f> features1,features2;
	// detect the features
	cv::goodFeaturesToTrack(gray, // the image 
		features1,   // the output detected features
		max_count,  // the maximum number of features 
		qlevel,     // quality level
		minDist);   // min distance between two features

	// 2. track features
	cv::calcOpticalFlowPyrLK(gray, pre_gray, // 2 consecutive images
		features1, // input point position in first image
		features2, // output point postion in the second image
		status,    // tracking success
		err);      // tracking error

	int k=0;

	for( int i= 0; i < features1.size(); i++ ) 
	{

		// do we keep this point?
		if (status[i] == 1) 
		{

			//m_features.data[(int)m_points[0][i].x+(int)m_points[0][i].y*m_oImgSize.width] = 0xff;
			// keep this point in vector
			features1[k] = features1[i];
			features2[k++] = features2[i];
		}
	}
	features1.resize(k);
	features2.resize(k);

	std::vector<uchar> inliers(features1.size());
	homography= cv::findHomography(
		cv::Mat(features1), // corresponding
		cv::Mat(features2), // points
		inliers, // outputted inliers matches
		CV_RANSAC, // RANSAC method
		0.1); // max distance to reprojection point
}
void TestFlow()
{
	DenseOpticalFlowProvier* DOFP = new GpuDenseOptialFlow();
	cv::Mat preImg = cv::imread("..//ptz//input3//in000289.jpg");
	cv::Mat curImg = cv::imread("..//ptz//input3//in000290.jpg");
	cv::cvtColor(preImg,preImg,CV_BGR2GRAY);
	cv::cvtColor(curImg,curImg,CV_BGR2GRAY);
	cv::Mat preMsk = cv::imread("..//result//subsensem//ptz//input3//wap//bin000289.png");
	cv::Mat curMsk = cv::imread("..//result//subsensem//ptz//input3//warp//bin000290.png");
	cv::cvtColor(preMsk,preMsk,CV_BGR2GRAY);
	cv::cvtColor(curMsk,curMsk,CV_BGR2GRAY);
	
	cv::Mat homography;
	GetHomography(curImg,preImg,homography);
	std::vector<cv::Point2f> curPts,prevPts;
	int width = curImg.cols;
	int height = curImg.rows;
	for(int i=0; i<width; i++)
	{
		for(int j=0; j<height; j++)
		{
			int idx = i+j*width;
			if (curMsk.data[idx] == 0xff)
				curPts.push_back(cv::Point2f(i,j));
		}
	}
	std::vector<uchar> status; // status of tracked features
	std::vector<float> err;    // error in tracking
	// 2. track features
	cv::calcOpticalFlowPyrLK(curImg, preImg, // 2 consecutive images
		curPts, // input point position in first image
		prevPts, // output point postion in the second image
		status,    // tracking success
		err);      // tracking error
	cv::Mat mask(height,width,CV_32F);
	mask = cv::Scalar(0);
	double* data = (double*)homography.data;
	double maxD(0),minD(1e10);

	for(int i=0; i<status.size(); i++)
	{
		if (status[i] == 1)
		{
			int x = (int)(curPts[i].x);
			int y = (int)(curPts[i].y);
			float wx = data[0]*x + data[1]*y + data[2];
			float wy = data[3]*x + data[4]*y + data[5];
			float w = data[6]*x + data[7]*y + data[8];
			wx /= w;
			wy /= w;
			int px = (int)(prevPts[i].x);
			int py = (int)(prevPts[i].y);
			int idx = py*width + px;
			float d = abs(px-wx) + abs(py-wy);
			
			if (d>maxD)
				maxD = d;
			if (d<minD)
				minD = d;
			float * ptr = (float*)(mask.data + idx*4);
			*ptr= d;
		}

	}

	cv::Mat B;
	mask.convertTo(B,CV_8U,255.0/(maxD-minD),0);
	cv::imwrite("tracked.jpg",B);
}

void SuperpixelFlow(const cv::Mat& sgray, const cv::Mat& tgray,int step, int spSize, const SLICClusterCenter* centers, cv::Mat& flow)
{
	std::vector<cv::Point2f> features0,features1;
	std::vector<uchar> status;
	std::vector<float> err;
	int spWidth = (sgray.cols+step-1)/step;
	int spHeight = (sgray.rows+step-1)/step;
	flow.create(spHeight,spWidth,CV_32FC2);
	flow = cv::Scalar(0);
	for(int i=0; i<spSize; i++)
	{
		features0.push_back(cv::Point2f(centers[i].xy.x,centers[i].xy.y));
	}
	cv::calcOpticalFlowPyrLK(sgray,tgray,features0,features1,status,err);

	int k=0; 
	for(int i=0; i<spHeight; i++)
	{
		float2 * ptr = flow.ptr<float2>(i);
		for(int j=0; j<spWidth; j++)
		{
			int idx = j + i*spWidth;
			if (status[idx] == 1)
			{
				ptr[j].x = features1[idx].x - features0[idx].x;
				ptr[j].y = features1[idx].y - features0[idx].y;
				k++;
			}
		}
	}
	std::cout<<"tracking succeeded "<<k<<" total "<<spSize<<std::endl;
}
void SuperpixelFlowToPixelFlow(const int* labels, const SLICClusterCenter* centers, const cv::Mat& sflow, int spSize, int step, int width, int height, cv::Mat& flow)
{
	
	flow.create(height,width,CV_32FC2);
	flow = cv::Scalar(0);
	for(int i=0; i<spSize; i++)
	{		
		int k = centers[i].xy.x;
		int j = centers[i].xy.y;
		if (centers[i].nPoints >0)			
		{
			
			//以原来的中心点为中心，step +2　为半径进行更新
			int radius = step;
			for (int x = k- radius; x<= k+radius; x++)
			{
				for(int y = j - radius; y<= j+radius; y++)
				{
					if  (x<0 || x>width-1 || y<0 || y> height-1)
						continue;
					int idx = x+y*width;
					
					if (labels[idx] == i )
					{		
						float2* fptr = (float2*)(flow.data + idx*8);
						*fptr = *((float2*)(sflow.data + i*8));
					}				
				}
			}
			
		}
	}
}

void SuperpixelMatching(const int* labels0, const SLICClusterCenter* centers0, const int* labels1, const SLICClusterCenter* centers1, int spSize, int spStep, int width, int height,
	const cv::Mat& mapX, const cv::Mat& mapY,std::vector<int> mathedId, cv::Mat& diff)
{
	diff.create(height,width,CV_32FC3);
	for(int i=0; i<spSize; i++)
	{		
		int k = centers0[i].xy.x;
		int j = centers0[i].xy.y;
		if (centers0[i].nPoints >0)			
		{
			float avgX(0),avgY(0);
			int n(0);
			//以原来的中心点为中心，step +2　为半径进行更新
			int radius = spStep;
			for (int x = k- radius; x<= k+radius; x++)
			{
				for(int y = j - radius; y<= j+radius; y++)
				{
					if  (x<0 || x>width-1 || y<0 || y> height-1)
						continue;
					int idx = x+y*width;
					
					if (labels0[idx] == i )
					{		
						float nx = *((float*)(mapX.data + idx*4));
						avgX += nx;
						float ny = *((float*)(mapY.data + idx*4));
						avgY += ny;
						n++;
					}				
				}
			}
			avgX /= n;
			avgY /= n;
			int iavgX = (int)(avgX +0.5);
			int iavgY = (int)(avgY + 0.5);
			int label = labels1[iavgX + iavgY*width];
			mathedId.push_back(label);
			float cdx = abs(centers0[i].rgb.x - centers1[label].rgb.x);
			float cdy = abs(centers0[i].rgb.y - centers1[label].rgb.y);
			float cdz = abs(centers0[i].rgb.z - centers1[label].rgb.z);
			float3 val = make_float3(cdx,cdy,cdz);
			k = centers1[label].xy.x;
			j = centers1[label].xy.y;
			for(int y = j - radius; y<= j+radius; y++)
			{
				if (y<0 || y> height-1)
					continue;
				float3* dPtr= diff.ptr<float3>(y);
				
				for (int x = k- radius; x<= k+radius; x++)
				{
					if  (x<0 || x>width-1)
						continue;
					int idx = x+y*width;
					if (labels1[idx] == label )
					{
						dPtr[x] = val;
					}
				}
			}
		}
	}
}
void TestSuperpixelFlow()
{
	SLIC aslic;	
	PictureHandler handler;
	
	int cols = 640;
	int rows = 480;
	int step = 5;
	ASAPWarping asap(cols,rows,8,1.0);

	int size = rows*cols;
	int num(0);
	GpuSuperpixel gs(cols,rows,step);
	cv::Mat simg,timg,wimg,sgray,tgray;
	cv::Mat img0,img1;
	img0 = cv::imread("..//moseg//people1//in000013.jpg");
	img1 = cv::imread("..//moseg//people1//in000012.jpg");
	
	std::vector<cv::Point2f> features0,features1;
	std::vector<uchar> status;
	std::vector<float> err;
	
	cv::cvtColor(img0,sgray,CV_BGR2GRAY);
	cv::cvtColor(img1,tgray,CV_BGR2GRAY);
	cv::cvtColor(img0,simg,CV_BGR2BGRA);
	cv::cvtColor(img1,timg,CV_BGR2BGRA);

	
	SLICClusterCenter* centers0(NULL),*centers1(NULL);
	int * labels0(NULL), * labels1(NULL);
	labels0 = new int[size];
	labels1 = new int[size];
	int spSize = ((rows+step-1)/step) * ((cols+step-1)/step);
	centers0 = new SLICClusterCenter[spSize];
	centers1 = new SLICClusterCenter[spSize];

	gs.Superpixel(simg,num,labels0,centers0);
	gs.Superpixel(timg,num,labels1,centers1);

	KLTFeaturesMatching(sgray,tgray,features0,features1);
	cv::Mat homography;
	FeaturePointsRefineRANSAC(features0,features1,homography);
	asap.SetControlPts(features0,features1);
	asap.Solve();
	asap.Warp(simg,wimg);
	std::vector<int> matchedId;
	cv::Mat diffMat;
	SuperpixelMatching(labels0,centers0,labels1,centers1,spSize,step,cols,rows,asap.getMapX(),asap.getMapY(),matchedId,diffMat);
	cv::imshow("matched err", diffMat);
	cv::waitKey();
	/*cv::Mat spFlow,flow,flowField,pflowField;
	std::vector<cv::Mat> flows(2);
	SuperpixelFlow(sgray,tgray,step,spSize,centers0,spFlow);
	cv::split(spFlow,flows);
	getFlowField(flows[0],flows[1],flowField);
	cv::imshow("superpixel flow",flowField);
	SuperpixelFlowToPixelFlow(labels0,centers0,spFlow,spSize,step,cols,rows,flow);
	cv::split(flow,flows);
	getFlowField(flows[0],flows[1],pflowField);
	cv::imshow("pixel flow",pflowField);
	cv::waitKey();*/


	//for(int i=0; i<spSize; i++)
	//{
	//	features0.push_back(cv::Point2f(centers0[i].xy.x,centers0[i].xy.y));
	//}
	//cv::calcOpticalFlowPyrLK(sgray,tgray,features0,features1,status,err);
	////跟踪成功的label
	//std::vector<int> mlabels0;
	//int k=0; 
	//for(int i=0; i<spSize; i++)
	//{
	//	if (status[i] ==1)
	//	{
	//		mlabels0.push_back(i);
	//		features0[k] = features0[i];
	//		features1[k++] = features1[i];
	//		
	//	}
	//}
	//asap.SetControlPts(features0,features1);
	//asap.Solve();
	//asap.Warp(img0,wimg);
	//cv::Mat mapX = asap.getMapX();
	//cv::Mat mapY = asap.getMapY();
	//int radius = 2*step;
	//cv::RNG rgn =  cv::theRNG();
	//for( int i=0; i<k; i++)
	//{
	//	
	//	int label0 = mlabels0[i];
	//	int ix1 = (int)(features1[i].x+0.5);
	//	int iy1 = (int)(features1[i].y+0.5);
	//	int idx = ix1+ iy1*cols;
	//	int ix0 = (int)(features0[i].x+0.5);
	//	int iy0 = (int)(features0[i].y+0.5);
	//	int idx0 = ix0+ iy0*cols;

	//	float dx = features1[i].x - mapX.at<float>(idx0);
	//	float dy = features1[i].y - mapY.at<float>(idx0);
	//	float dis = abs(dx) + abs(dy);

	//	int label1 = labels1[idx];
	//	float4 avg0 = centers0[label0].rgb;
	//	float4 avg1 = centers1[label1].rgb;
	//	uchar diff = (abs(avg0.x-avg0.y) + abs(avg0.y - avg1.y) + abs(avg0.z-avg1.z) + abs(avg0.w-avg1.w))/4;
	//	cv::Scalar color = cv::Scalar(diff,diff,diff,255);
	//	if (dis < 2)
	//		color = cv::Scalar(255,0,0,255);

	//	
	//	for(int m = -radius; m<= radius; m++)
	//	{
	//		for(int n = -radius; n<= radius; n++)
	//		{
	//			int ix = features0[i].x+m;
	//			int iy = features0[i].y+n;
	//			
	//		
	//			if (ix >=0 && ix < cols && iy >=0 && iy<rows)
	//			{	
	//				int idx0 = ix + iy*cols;
	//				if (labels0[idx0] == label0)
	//				{						
	//					simg.at<Vec4b>(iy,ix) = color;
	//				}
	//			}
	//			
	//			ix = centers1[label1].xy.x+m;
	//			iy = centers1[label1].xy.y+n;
	//			if (ix >=0 && ix < cols && iy >=0 && iy<rows)
	//			{
	//				int idx1 = ix + iy*cols;
	//				if (labels1[idx1] == label1)
	//				{
	//					timg.at<Vec4b>(iy,ix) = color;
	//				}
	//			}
	//		}
	//	}
	//	
	//		
	//}

	//unsigned int* idata = new unsigned[size];
	//memcpy(idata,simg.data,size*4);
	//aslic.DrawContoursAroundSegments(idata, labels0, simg.cols,simg.rows,0x00ff00);
	//aslic.SaveSuperpixelLabels(labels0,cols,rows,std::string("labels0.txt"),std::string(".\\"));
	//handler.SavePicture(idata,simg.cols,simg.rows,std::string("GpuSp0.jpg"),std::string(".\\"));
	//memcpy(idata,timg.data,size*4);
	//aslic.DrawContoursAroundSegments(idata, labels1, simg.cols,simg.rows,0x00ff00);
	//aslic.SaveSuperpixelLabels(labels1,cols,rows,std::string("labels1.txt"),std::string(".\\"));
	//handler.SavePicture(idata,simg.cols,simg.rows,std::string("GpuSp1.jpg"),std::string(".\\"));

	//cv::imshow("s",simg);
	//cv::imshow("t", timg);
	//cv::waitKey(0);
	//if(labels0 != NULL)
	//{
	//	delete[] labels0;
	//	delete[] labels1;
	//	delete[] centers0;
	//	delete[] centers1;
	//	delete[] idata;
	//	idata = NULL;
	//	centers0 = NULL;
	//	centers1 = NULL;
	//	labels0 = NULL;
	//	labels1 = NULL;
	//}
}
void TCMRFOptimization()
{
	/*TestFlow();
	return;*/
	using namespace std;
	char imgFileName[150];
	char maskFileName[150];
	char resultFileName[150];
	int cols = 320;
	int rows = 240;
	GpuSuperpixel gs(cols,rows,5);
	MRFOptimize optimizer(cols,rows,5);
	nih::Timer timer;
	timer.start();
	int start = 300;
	int end = 330;
	std::vector<cv::Mat> imgs;
	std::vector<cv::Mat> masks;
	cv::Mat curImg,prevImg,mask,prevMask,resultImg,gray,preGray;

	cv::Mat flow;
	//DenseOpticalFlowProvier* DOFP = new GpuDenseOptialFlow();
	//DenseOpticalFlowProvier* DOFP = new SFDenseOptialFlow();
	DenseOpticalFlowProvier* DOFP = new EPPMDenseOptialFlow();
	for(int i=start; i<=end;i++)
	{
		sprintf(imgFileName,"..\\ptz\\input3\\in%06d.jpg",i);		
		curImg = cv::imread(imgFileName);
		imgs.push_back(curImg.clone());
		sprintf(maskFileName,"..\\result\\subsensem\\ptz\\input3\\bin%06d.png",i);
		curImg = cv::imread(maskFileName);
		cv::cvtColor(curImg,curImg,CV_BGR2GRAY);
		masks.push_back(curImg.clone());		
	}
	for (int i= 1; i<end-start+1; i++)
	{
		curImg = imgs[i];
		prevImg = imgs[i-1];
		prevMask = masks[i-1];
		cv::cvtColor(curImg,gray,CV_BGR2GRAY);
		cv::cvtColor(prevImg,preGray,CV_BGR2GRAY);
		mask = masks[i];
		cv::Mat homography;
		GetHomography(gray,preGray,homography);
		DOFP->DenseOpticalFlow(gray,preGray,flow);
		//WriteFlowFile(flow,"flow.flo");
		//ReadFlowFile(flow,"ptz0_87.flo");
		optimizer.Optimize(&gs,curImg,mask,prevMask,flow,homography,resultImg);
		sprintf(resultFileName,"..\\result\\SubsenseMMRF\\ptz\\input3\\bin%06d.png",i);
		cv::imwrite(resultFileName,resultImg);
	}
	timer.stop();
	std::cout<<(end-start+1)/timer.seconds()<<" fps\n";
	delete DOFP;
}
