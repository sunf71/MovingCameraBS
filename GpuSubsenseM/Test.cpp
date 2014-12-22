#include "Test.h"
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
	//cv::resize(img,img,cv::Size(16,16));
	uchar4* imgData = new uchar4[img.rows*img.cols];
	unsigned int* idata = new unsigned int[img.rows*img.cols];
	for(int i=0; i< img.cols; i++)
	{
		
		for(int j=0; j<img.rows; j++)
		{
			int idx = img.step[0]*j + img.step[1]*i;
			imgData[i + j*img.cols].x = img.data[idx];
			imgData[i + j*img.cols].y = img.data[idx+ img.elemSize1()];
			imgData[i + j*img.cols].z = img.data[idx+2*img.elemSize1()];
			imgData[i + j*img.cols].w = img.data[idx+3*img.elemSize1()];
			unsigned char tmp[4];
			for(int k=0; k<4; k++)
				tmp[k] = img.data[img.step[0]*j + img.step[1]*i + img.elemSize1()*k];
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
	int start = 1;
	int end = 19;
	for(int i=start; i<=end;i++)
	{
		char name[50];
		sprintf(name,"..\\moseg\\people1\\in%06d.jpg",i);
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
	
	cv::gpu::GpuMat dCurImg(curImg);
	cv::gpu::GpuMat dPrevImg(prevImg);
	cv::gpu::cvtColor(dCurImg,dCurImg,CV_BGR2GRAY);
	cv::gpu::cvtColor(dPrevImg,dPrevImg,CV_BGR2GRAY);
	cv::gpu::GpuMat du,dv;
	cv::gpu::PyrLKOpticalFlow dPyrLK;

	dPyrLK.dense(dCurImg,dPrevImg,du,dv);
	
	std::vector<cv::Mat> flows(2);
	du.download(flows[0]);
	dv.download(flows[1]);
	cv::Mat flowField;
	getFlowField(flows[0],flows[1],flowField);
	cv::imwrite("flow.jpg",flowField);
	
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
	//cv::calcOpticalFlowSF(img0,img1,flow,3,2,15);
	cv::calcOpticalFlowFarneback(img0,img1,flow,0.5, 3, 15, 3, 5, 1.2, 0);
	//cv::Mat flowField;
	//cv::Mat flows[2];
	//cv::split(flow,flows);
	//getFlowField(flows[0],flows[1],flowField);
	//cv::imwrite("flow_Farnback.jpg",flowField);
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
void TCMRFOptimization()
{
	using namespace std;
	char imgFileName[150];
	char maskFileName[150];
	char resultFileName[150];
	int cols = 704;
	int rows = 480;
	GpuSuperpixel gs(cols,rows,5);
	MRFOptimize optimizer(cols,rows,5);
	nih::Timer timer;
	timer.start();
	int start = 7;
	int end = 8;
	std::vector<cv::Mat> imgs;
	std::vector<cv::Mat> masks;
	cv::Mat curImg,prevImg,mask,prevMask,resultImg,gray,preGray;

	cv::Mat flow;
	//DenseOpticalFlowProvier* DOFP = new GpuDenseOptialFlow();
	//DenseOpticalFlowProvier* DOFP = new SFDenseOptialFlow();
	DenseOpticalFlowProvier* DOFP = new FarnebackDenseOptialFlow();
	for(int i=start; i<=end;i++)
	{
		sprintf(imgFileName,"..\\ptz\\input0\\in%06d.jpg",i);		
		curImg = cv::imread(imgFileName);
		imgs.push_back(curImg.clone());
		sprintf(maskFileName,"..\\result\\subsensex\\ptz\\input0\\bin%06d.png",i);
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
		optimizer.Optimize(&gs,curImg,mask,prevMask,flow,homography,resultImg);
		sprintf(resultFileName,"..\\result\\SubsenseMMRF\\ptz\\input0\\bin%06d.png",i);
		cv::imwrite(resultFileName,resultImg);
	}
	timer.stop();
	std::cout<<(end-start+1)/timer.seconds()<<" fps\n";
	delete DOFP;
}