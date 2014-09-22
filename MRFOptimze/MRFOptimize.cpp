#include "MRFOptimize.h"
#include "GCoptimization.h"
#include "SLIC.h"
#include "ComSuperpixel.h"
#include "PictureHandler.h"
#include "timer.h"
bool compare(const Point2i& p1, const Point2i& p2)
{
	if (p1.first==p2.first)
		return p1.second<p2.second;
	else
		return p1.first < p2.first;
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
			int idxC = idx*4;
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
	/*	for(int p=0; p<superpixels[i].pixels.size(); p++)
		{
			int idx = superpixels[i].pixels[p].first + superpixels[i].pixels[p].second*width;
			psImg[idx] = uchar(superpixels[i].ps*255);
			avgImg[idx] = uchar(superpixels[i].avgColor);
		}*/
	}
	/*cv::imwrite("prob.jpg",psMat);
	cv::imwrite("avg.jpg",avgMat);*/
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
	int* data = new int[num_pixels*num_labels];

	for(int i=0; i<num_pixels; i++)
	{
		for(int j=0; j<num_labels; j++)
		{
			float d = min(1.0f,theta*spPtr[i].ps*2);
			d = max(1e-20f,d);
			float d1 = -log(d)*j;
			float d2 =  - log(1-d)*(1-j);
			data[i*num_labels + j] =(int)5*(d1+d2);
		}
	}
	// next set up the array for smooth costs
	int *smooth = new int[num_labels*num_labels];
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
					gc->setNeighbors(i,spPtr[i].neighbors[j]->idx,(int)(energy));
				}
			}
		}
		/*file.close();
		printf("\nBefore optimization energy is %d",gc->compute_energy());
		printf("\nBefore optimization  data energy is %d",gc->giveDataEnergy());
		printf("\nBefore optimization smooth energy is %d",gc->giveSmoothEnergy());*/
		gc->expansion(10);// run expansion for 2 iterations. For swap use gc->swap(num_iterations);
		/*printf("\nAfter optimization energy is %d",gc->compute_energy());
		printf("\nAfter optimization  data energy is %d",gc->giveDataEnergy());
		printf("\nAfter optimization smooth energy is %d",gc->giveSmoothEnergy());*/

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
	//int* clabels = new int[sz];
	
	int numlabels(0);
	ComSuperpixel CS;
	//CS.Superixel(idata,width,height,7000,0.9,labels);
#ifdef REPORT
	nih::Timer timer;
	timer.start();
#endif
	CS.Superixel(idata,width,height,5,0.9,numlabels,labels);
#ifdef REPORT
	timer.stop();
	std::cout<<"SLIC SuperPixel "<<timer.seconds()<<std::endl;
#endif
	SLIC aslic;
	aslic.DrawContoursAroundSegments(idata, labels, width, height,0x00ff00);
	PictureHandler handler;
	handler.SavePicture(idata,width,height,std::string("mysuper.jpg"),std::string(".\\"));
	//delete[] labels;
	//return;
//#ifdef REPORT
//	nih::Timer timer;
//	timer.start();
//#endif
//	//SLIC slic;
//	//slic.PerformSLICO_ForGivenK(idata, width, height, labels, numlabels, 2000, 20);//for a given number K of superpixels
//#ifdef REPORT
//	timer.stop();
//	std::cout<<"SLIC SuperPixel "<<timer.seconds()<<std::endl;
//#endif
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
			if (i < spPtr[i].neighbors[j]->idx)
			{

				avgE += abs(spPtr[i].avgColor-spPtr[i].neighbors[j]->avgColor);
				count++;
			}

		}
	}
	avgE /= count;
	
	avgE = 1/(2*avgE);
	//std::cout<<"avg e "<<avgE<<std::endl;
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