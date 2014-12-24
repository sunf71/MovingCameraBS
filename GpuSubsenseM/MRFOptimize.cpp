#include "MRFOptimize.h"
#include "GCoptimization.h"
#include "SLIC.h"
#include "GpuSuperpixel.h"
#include "ComSuperpixel.h"
#include "PictureHandler.h"
#include "timer.h"
#include "GpuTimer.h"

bool compare(const Point2i& p1, const Point2i& p2)
{
	if (p1.first==p2.first)
		return p1.second<p2.second;
	else
		return p1.first < p2.first;
}
void MRFOptimize::Init()
{
	m_QSIZE = 1024*1024;
	int size = m_width*m_height;
	m_spPtr = new SuperPixel[m_width*m_height];
	m_visited = new bool[size];
	m_stack = new Point2i[m_QSIZE];
	m_gWidth = ((m_width+ m_step-1) / m_step);
	m_gHeight = ((m_height + m_step -1) / m_step);
	m_nPixel =  m_gWidth*m_gHeight;
	m_centers = new SLICClusterCenter[m_nPixel];
	m_data = new int[m_nPixel*2];
	m_smooth = new int[4];
	for ( int l1 = 0; l1 < 2; l1++ )
		for (int l2 = 0; l2 < 2; l2++ )
			m_smooth[l1+l2*2] = abs(l1-l2);
	/*m_gc = new GCoptimizationGeneralGraph(m_nPixel,2);*/
	//m_gc->setSmoothCost(m_smooth);
	m_result= new int[m_nPixel];   // stores result of optimization
	m_imgData = new uchar4[size];
	m_idata = new unsigned int[size];
	m_labels = new int[size];
	m_neighbor.resize(m_nPixel);
	int xStep = ((m_width+ m_step-1) / m_step);
	for(int i=0; i<m_nPixel; i++)
	{

		if (i-1>=0)
		{
			m_neighbor[i].push_back(i-1);
			if(i+xStep-1<m_nPixel)
				m_neighbor[i].push_back(i+xStep-1);
			if (i-xStep-1>=0)
				m_neighbor[i].push_back(i-xStep-1);
		}
		if(i+1<m_nPixel)
		{
			m_neighbor[i].push_back(i+1);
			if(i+xStep+1<m_nPixel)
				m_neighbor[i].push_back(i+xStep+1);
			if (i-xStep+1>=0)
				m_neighbor[i].push_back(i-xStep+1);
		}
		
		if (i-xStep>=0)
			m_neighbor[i].push_back(i-xStep);		
		
		if(i+xStep<m_nPixel)
			m_neighbor[i].push_back(i+xStep);
		
	}
	
	
}
	

void MRFOptimize::Release()
{
	delete[] m_visited;
	delete[] m_stack;

	delete[] m_data;
	delete[] m_smooth;
	delete[] m_imgData;
	delete[] m_idata;	
	delete[] m_result;
	delete[] m_labels;
	delete[] m_centers;
	delete[] m_spPtr;
	
}


//save superpixels to an array with neighbor and pixels info
void MRFOptimize::GetSegment2DArray(SuperPixel *& superpixels, size_t & spSize, const int* lables, const int width,const int height)
{
	size_t size = width*height;	
	SuperPixelMap map;	
	memset(m_visited,0,sizeof(bool)*size);
	//size of superpixels;
	spSize = 0;
	const int dx[8] = {-1, -1,  0,  1, 1, 1, 0, -1};
	const int dy[8] = { 0, -1, -1, -1, 0, 1, 1,  1};	

	int ptr = -1;
	m_stack[++ptr] = Point2i(0,0);
	while(ptr >= 0)
	{
		//std::cout<<spSize<<std::endl;
		if (ptr>m_QSIZE)
			std::cout<<ptr<<std::endl;
		Point2i pt = m_stack[ptr--];
		int idx = pt.first + pt.second*width;
		if (!m_visited[idx])
		{
			m_visited[idx] = true;
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
				if (!m_visited[nid])
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
					m_stack[++ptr] = np;					
				}
			}
		}
	}
}

void MRFOptimize::ComputeAvgColor(SuperPixel* superpixels, size_t spSize, const int width, const int height,  const unsigned int* imgData, const unsigned char* maskData)
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
		/*for(int p=0; p<superpixels[i].pixels.size(); p++)
		{
			int idx = superpixels[i].pixels[p].first + superpixels[i].pixels[p].second*width;
			psImg[idx] = uchar(superpixels[i].ps*255);
			avgImg[idx] = uchar(superpixels[i].avgColor);
		}*/
	}
	/*cv::imwrite("prob.jpg",psMat);
	cv::imwrite("avg.jpg",avgMat);*/
}
void MRFOptimize::MaxFlowOptimize(SuperPixel* spPtr, int num_pixels,float beta, int num_labels,const int width, const int height,int *result)
{
	size_t num_edges = 0;
	for(int i=0; i<num_pixels; i++)
		num_edges += m_neighbor[i].size();
	num_edges /= 2;

	typedef Graph<float,float,float> GraphType;
	GraphType *g = new GraphType(/*estimated # of nodes*/ num_pixels, /*estimated # of edges*/ num_edges); 
	float theta = 0.5;
	for(int i=0; i<num_pixels; i++)
	{
		g->add_node();
		
		float d = min(1.0f,spPtr[i].distance);
		float dis = spPtr[i].distance;
		float dd1 = exp(-dis/theta);
		float dd2 = 1-dd1;
		
		d = max(1e-20f,d);
		float d1 = -log(d);		
		float d2 =  - log(1-d);
		//std::cout<<dd1<<std::endl;
		g->add_tweights(i,d1,d2);
	}
	for(int i=0; i<num_pixels; i++)
	{

		for(int j=0; j<m_neighbor[i].size(); j++)
		{				
			if (i>spPtr[m_neighbor[i][j]].idx)
			{
				float energy = (m_lmd1+m_lmd2*exp(-beta*abs(spPtr[i].avgColor-spPtr[m_neighbor[i][j]].avgColor)));
				//file<<energy<<std::endl;
				g->add_edge(i,m_neighbor[i][j],0,0);
			}
		}
	}

	float flow = g -> maxflow();
	for ( int  i = 0; i < num_pixels; i++ )
		result[i] = g->what_segment(i) == GraphType::SINK ? 0x1 : 0;
}

void MRFOptimize::GraphCutOptimize(SuperPixel* spPtr, int num_pixels,float beta, int num_labels,const int width, const int height,int *result)
{
	// first set up the array for data costs
	//nih::Timer timer;
	//timer.start();
	/*std::ofstream dfile("dtenergy.txt");*/
	for(int i=0; i<num_pixels; i++)
	{
		for(int j=0; j<num_labels; j++)
		{
			float d = min(1.0f,m_theta*spPtr[i].ps*2);
			d = max(1e-20f,d);
			float d1 = -log(d)*j;
			float d2 =  - log(1-d)*(1-j);
			m_data[i*num_labels + j] =(int)5*(d1+d2);
			/*dfile<<m_data[i*num_labels + j]<<std::endl;*/
		}
	}
	//timer.stop();
	//std::cout<<"set data cost "<<timer.seconds()*1000<<std::endl;
	/*dfile.close();*/
	// next set up the array for smooth costs



	try{
		GCoptimizationGeneralGraph *gc = new GCoptimizationGeneralGraph(num_pixels,num_labels);
		gc->setDataCost(m_data);
		//gc->setSmoothCost(&smoothFn);
		gc->setSmoothCost(m_smooth);
		//std::ofstream file("out.txt");
		// now set up a grid neighborhood system
		for(int i=0; i<num_pixels; i++)
		{
			//std::cout<<i<<std::endl;
			for(int j=0; j<m_neighbor[i].size(); j++)
			{				
				if (i>spPtr[m_neighbor[i][j]].idx)
				{
					float energy = (m_lmd1+m_lmd2*exp(-beta*abs(spPtr[i].avgColor-spPtr[m_neighbor[i][j]].avgColor)));
					//file<<energy<<std::endl;
					gc->setNeighbors(i,m_neighbor[i][j],(int)(energy));
				}
			}
			//for(int j=0; j<spPtr[i].neighbors.size(); j++)
			//{				
			//	if (i>spPtr[i].neighbors[j]->idx)
			//	{
			//		float energy = (m_lmd1+m_lmd2*exp(-beta*abs(spPtr[i].avgColor-spPtr[i].neighbors[j]->avgColor)));
			//		//file<<energy<<std::endl;
			//		gc->setNeighbors(i,spPtr[i].neighbors[j]->idx,(int)(energy));
			//	}
			//}
		}
		//timer.stop();
		//std::cout<<"set setNeighbors cost "<<timer.seconds()*1000<<std::endl;
		//timer.start();
		/*file.close();*/
	/*	printf("\nBefore optimization energy is %d",gc->compute_energy());
		printf("\nBefore optimization  data energy is %d",gc->giveDataEnergy());
		printf("\nBefore optimization smooth energy is %d",gc->giveSmoothEnergy());*/
		gc->expansion(10);// run expansion for 2 iterations. For swap use gc->swap(num_iterations);
		/*printf("\nAfter optimization energy is %d",gc->compute_energy());
		printf("\nAfter optimization  data energy is %d",gc->giveDataEnergy());
		printf("\nAfter optimization smooth energy is %d",gc->giveSmoothEnergy());*/
		/*timer.stop();
		std::cout<<"expansion "<<timer.seconds()*1000<<std::endl;*/
		for ( int  i = 0; i < num_pixels; i++ )
			result[i] = gc->whatLabel(i);
		delete gc;
	}
	catch (GCException e){
		e.Report();
	}
}
void NeighborsImage(SuperPixel* sp, int * labels,int nPixel, int width, int height,std::vector<std::vector<int>>& mNeighbors)
{
	int id = 4;
	cv::Mat img(height,width,CV_8U);
	img = cv::Scalar(0);
	uchar* imgPtr = img.data;
	for(int i= 0; i<width; i++)
	{
		for(int j=0; j<height; j++)
		{
			int idx = i+j*width;
			if (labels[idx] == sp[id].lable)
				imgPtr[idx] = 128;

		}	
	}
	for(int k=0; k<mNeighbors[id].size(); k++)
	{
		for(int i= 0; i<width; i++)
		{
			for(int j=0; j<height; j++)
			{
				int idx = i+j*width;
				if (labels[idx] == mNeighbors[id][k])
					imgPtr[idx] = 0xff;
				
			}	
		}
	}
	cv::imwrite("neighbour.jpg",img);
}
void ProbImage(SuperPixel* sp, int * labels,int nPixel, int width, int height)
{
	cv::Mat img(height,width,CV_8U);
	uchar* imgPtr = img.data;
	cv::Mat pimg(height,width,CV_8U);
	uchar* pimgPtr = pimg.data;

	for(int i= 0; i<width; i++)
	{
		for(int j=0; j<height; j++)
		{
			int idx = i+j*width;
			imgPtr[idx] = sp[labels[idx]].avgColor;
			pimgPtr[idx] = sp[labels[idx]].ps*255;
		}	
	}
	cv::imwrite("avg.jpg",img);
	cv::imwrite("prob.jpg",pimg);
}
//用MRF对前景结果进行优化
void MRFOptimize::Optimize(GpuSuperpixel* GS, const string& originalImgName, const string& maskImgName, const string& resultImgName)
{
#ifdef REPORT
	nih::Timer timer;
	nih::Timer timer0;
	timer0.start();
	timer.start();
#endif
	m_spPtr = new SuperPixel[m_width*m_height];
	//superpixel
	cv::Mat img = cv::imread(originalImgName);		
	cv::Mat continuousRGBA(img.size(), CV_8UC4, m_idata);	
	cv::Mat FImg(img.size(), CV_32FC4, m_imgData);
	cv::cvtColor(img,continuousRGBA,CV_BGR2BGRA,4);
	continuousRGBA.convertTo(FImg,CV_32FC4);


	cv::Mat maskImg = cv::imread(maskImgName);
	cv::cvtColor(maskImg,maskImg,CV_BGR2GRAY);
	maskImg.convertTo(maskImg,CV_8U);
	const unsigned char* maskImgData = maskImg.data;
#ifdef REPORT
	timer.stop();
	std::cout<<"read image  "<<timer.seconds()*1000 <<"ms"<<std::endl;
#endif

	int numlabels(0);

#ifdef REPORT
	GpuTimer gtimer;
	gtimer.Start();
#endif
	GS->Superpixel(m_imgData,numlabels,m_labels,m_centers);
	
	//cv::Mat labelImg(m_height,m_width,CV_8U);
	//labelImg = cv::Scalar(0);
	//uchar* ptr = labelImg.data;
	//std::ofstream labelF("label.tmp");
	//for(int y=0; y<m_height; y++)
	//{
	//	for(int x=0; x<m_width; x++)
	//	{
	//		int idx = x+ y*m_width;
	//		labelF<<m_labels[idx]<<" ";			
	//	}
	//	labelF<<"\n ";
	//}
	//labelF.close();
	//cv::imwrite("label.jpg",labelImg);
	//ComSuperpixel cs;
	//cs.Superpixel(m_idata,m_width,m_height,5,0.9,numlabels,m_labels);
#ifdef REPORT
	gtimer.Stop();
	std::cout<<"GPU SuperPixel "<<gtimer.Elapsed()<<"ms"<<std::endl;
#endif
	/*std::ofstream fileout("out.tmp.");
	for(int i=0; i<width; i++)
	{
	for(int j=0; j<height; j++)
	{
	fileout<<labels[i+j*width]<<"\t ";
	}
	fileout<<"\n";
	}
	fileout.close();*/

	/*SLIC aslic;
	aslic.DrawContoursAroundSegments(m_idata, m_labels, m_width, m_height,0x00ff00);
	PictureHandler handler;
	handler.SavePicture(m_idata,m_width,m_height,std::string("mysuper.jpg"),std::string(".\\"));*/
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
//#ifdef REPORT
//	timer.start();
//#endif

	/*size_t spSize(0);
	GetSegment2DArray(m_spPtr,spSize,m_labels,m_width,m_height);
	cv::Mat  sp(m_height,m_width,CV_8UC3);
	for(int i=0; i<spSize; i++)
	{
		cv::Vec3b color(rand()%255,rand()%255,rand()%255);
		for(int j=0; j<m_spPtr[i].pixels.size(); j++)
		{

			sp.at<cv::Vec3b>(m_spPtr[i].pixels[j].second,m_spPtr[i].pixels[j].first) = color;
		}
	}	
	cv::imwrite("sp.jpg",sp);*/
	
//#ifdef REPORT
//	timer.stop();
//	std::cout<<"GetSegment2DArray  "<<timer.seconds()*1000<<"ms"<<std::endl;
//#endif
//
#ifdef REPORT

	timer.start();
#endif
	//ComputeAvgColor(m_spPtr,spSize,m_width,m_height,m_idata,maskImgData);
	GetSuperpixels(maskImgData);
	//ProbImage(m_spPtr,m_labels,m_nPixel,m_width,m_height);
	//std::ofstream outFile("nb.tmp");
	//for(int id = 0; id<m_nPixel; id++)
	//{
	//	outFile<<id<<" ";
	//	for(int i=0; i<m_spPtr[id].neighbors.size();i++)
	//	{
	//		outFile<<m_spPtr[id].neighbors[i]->lable<<" ";
	//	}
	//	outFile<<std::endl;
	//}
	//outFile.close();
	//NeighborsImage(m_spPtr,m_labels,m_nPixel,m_width,m_height);
#ifdef REPORT
	timer.stop();
	std::cout<<"GetSuperpixels  "<<timer.seconds()*1000<<"ms"<<std::endl;
#endif
#ifdef REPORT
	timer.start();
#endif
	float avgE = 0;
	size_t count = 0;	
	for(int i=0; i<m_nPixel; i++)
	{
		for (int j=0; j< m_neighbor[i].size(); j++)
		{
			if (m_centers[i].nPoints > 0 && m_centers[m_neighbor[i][j]].nPoints >0 )
			{				
				avgE += abs(m_spPtr[i].avgColor-m_spPtr[m_neighbor[i][j]].avgColor);
				count++;
			}
		}
		//for (int j=0; j< m_spPtr[i].neighbors.size(); j++)
		//{
		//	if (m_spPtr[i].lable < m_spPtr[i].neighbors[j]->lable)
		//	{

		//		avgE += abs(m_spPtr[i].avgColor-m_spPtr[i].neighbors[j]->avgColor);
		//		count++;
		//	}

		//}
	}
	avgE /= count;
	avgE = 1/(2*avgE);
	//std::cout<<"avg e "<<avgE<<std::endl;
#ifdef REPORT
	timer.stop();
	std::cout<<"ComputeAvgColor  "<<timer.seconds()*1000<<"ms"<<std::endl;
#endif

#ifdef REPORT

	timer.start();
#endif
	//GraphCutOptimize(m_spPtr,m_nPixel,avgE,2,m_width,m_height,m_result);
	//MaxFlowOptimize(m_spPtr,m_nPixel,avgE,2,m_width,m_height,m_result);
	GridCutOptimize(m_spPtr,m_nPixel,avgE,2,m_width,m_height,m_result);

#ifdef REPORT
	timer.stop();
	std::cout<<"GraphCutOptimize  "<<timer.seconds()*1000<<"ms"<<std::endl;
#endif

#ifdef REPORT	
	timer.start();
#endif
	cv::Mat rimg(m_height,m_width,CV_8U);
	rimg = cv::Scalar(0);
	unsigned char* imgPtr = rimg.data;
	for(int i=0; i<m_nPixel; i++)
	{
		if(m_result[i] == 1)			
		{
			/*for(int j=0; j<m_spPtr[i].pixels.size(); j++)
			{
			int idx = m_spPtr[i].pixels[j].first + m_spPtr[i].pixels[j].second*m_width;
			imgPtr[idx] = 0xff;
			}*/
			int k = m_centers[i].xy.x;
			int j = m_centers[i].xy.y;
			int radius = m_step+5;
			for (int x = k- radius; x<= k+radius; x++)
			{
				for(int y = j - radius; y<= j+radius; y++)
				{
					if  (x<0 || x>m_width-1 || y<0 || y> m_height-1)
						continue;
					int idx = x+y*m_width;
					if (m_labels[idx] == i)
					{					
						imgPtr[idx] = 0xff;
					}					
				}
			}
		}	
	}
	cv::imwrite(resultImgName,rimg);
#ifdef REPORT
	timer.stop();
	std::cout<<"imwrite  "<<timer.seconds()*1000<<"ms"<<std::endl;
#endif
	delete[] m_spPtr;

#ifdef REPORT
	timer0.stop();
	std::cout<<"optimize one frame  "<<timer0.seconds()*1000<<"ms"<<std::endl;
#endif
}
void MRFOptimize::Optimize(GpuSuperpixel* GS, cv::Mat& origImg, cv::Mat& maskImg, cv::Mat& featureMaskImg, cv::Mat& resultImg)
{
	#ifdef REPORT
	nih::Timer timer;
	nih::Timer timer0;
	timer0.start();
	timer.start();
#endif

	//superpixel		
	cv::Mat FImg(origImg.size(), CV_8UC4, m_imgData);
	cv::Mat continuousRGBA(origImg.size(), CV_8UC4, m_idata);
	cv::cvtColor(origImg,continuousRGBA,CV_BGR2BGRA);
	continuousRGBA.convertTo(FImg,CV_8UC4);

	const unsigned char* maskImgData = maskImg.data;
	const unsigned char* featureMaskData = featureMaskImg.data;
#ifdef REPORT
	timer.stop();
	std::cout<<"read image  "<<timer.seconds()*1000 <<"ms"<<std::endl;
#endif

	int numlabels(0);

#ifdef REPORT
	GpuTimer gtimer;
	gtimer.Start();
#endif
	GS->Superpixel(m_imgData,numlabels,m_labels,m_centers);
	
	//cv::Mat labelImg(m_height,m_width,CV_8U);
	//labelImg = cv::Scalar(0);
	//uchar* ptr = labelImg.data;
	//std::ofstream labelF("label.tmp");
	//for(int y=0; y<m_height; y++)
	//{
	//	for(int x=0; x<m_width; x++)
	//	{
	//		int idx = x+ y*m_width;
	//		labelF<<m_labels[idx]<<" ";			
	//	}
	//	labelF<<"\n ";
	//}
	//labelF.close();
	//cv::imwrite("label.jpg",labelImg);
	//ComSuperpixel cs;
	//cs.Superpixel(m_idata,m_width,m_height,5,0.9,numlabels,m_labels);
#ifdef REPORT
	gtimer.Stop();
	std::cout<<"GPU SuperPixel "<<gtimer.Elapsed()<<"ms"<<std::endl;
#endif
	/*std::ofstream fileout("out.tmp.");
	for(int i=0; i<width; i++)
	{
	for(int j=0; j<height; j++)
	{
	fileout<<labels[i+j*width]<<"\t ";
	}
	fileout<<"\n";
	}
	fileout.close();*/

	/*SLIC aslic;
	aslic.DrawContoursAroundSegments(m_idata, m_labels, m_width, m_height,0x00ff00);
	PictureHandler handler;
	handler.SavePicture(m_idata,m_width,m_height,std::string("mysuper.jpg"),std::string(".\\"));*/
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
//#ifdef REPORT
//	timer.start();
//#endif

	/*size_t spSize(0);
	GetSegment2DArray(m_spPtr,spSize,m_labels,m_width,m_height);
	cv::Mat  sp(m_height,m_width,CV_8UC3);
	for(int i=0; i<spSize; i++)
	{
		cv::Vec3b color(rand()%255,rand()%255,rand()%255);
		for(int j=0; j<m_spPtr[i].pixels.size(); j++)
		{

			sp.at<cv::Vec3b>(m_spPtr[i].pixels[j].second,m_spPtr[i].pixels[j].first) = color;
		}
	}	
	cv::imwrite("sp.jpg",sp);*/
	
//#ifdef REPORT
//	timer.stop();
//	std::cout<<"GetSegment2DArray  "<<timer.seconds()*1000<<"ms"<<std::endl;
//#endif
//
#ifdef REPORT

	timer.start();
#endif
	//ComputeAvgColor(m_spPtr,spSize,m_width,m_height,m_idata,maskImgData);
	GetSuperpixels(maskImgData,featureMaskData);
	//ProbImage(m_spPtr,m_labels,m_nPixel,m_width,m_height);
	//std::ofstream outFile("nb.tmp");
	//for(int id = 0; id<m_nPixel; id++)
	//{
	//	outFile<<id<<" ";
	//	for(int i=0; i<m_spPtr[id].neighbors.size();i++)
	//	{
	//		outFile<<m_spPtr[id].neighbors[i]->lable<<" ";
	//	}
	//	outFile<<std::endl;
	//}
	//outFile.close();
	//NeighborsImage(m_spPtr,m_labels,m_nPixel,m_width,m_height);
#ifdef REPORT
	timer.stop();
	std::cout<<"GetSuperpixels  "<<timer.seconds()*1000<<"ms"<<std::endl;
#endif
#ifdef REPORT
	timer.start();
#endif
	float avgE = 0;
	size_t count = 0;	
	for(int i=0; i<m_nPixel; i++)
	{
		for (int j=0; j< m_neighbor[i].size(); j++)
		{
			if (m_centers[i].nPoints > 0 && m_centers[m_neighbor[i][j]].nPoints >0 )
			{				
				avgE += abs(m_spPtr[i].avgColor-m_spPtr[m_neighbor[i][j]].avgColor);
				count++;
			}
		}
		//for (int j=0; j< m_spPtr[i].neighbors.size(); j++)
		//{
		//	if (m_spPtr[i].lable < m_spPtr[i].neighbors[j]->lable)
		//	{

		//		avgE += abs(m_spPtr[i].avgColor-m_spPtr[i].neighbors[j]->avgColor);
		//		count++;
		//	}

		//}
	}
	avgE /= count;
	avgE = 1/(2*avgE);
	//std::cout<<"avg e "<<avgE<<std::endl;
#ifdef REPORT
	timer.stop();
	std::cout<<"ComputeAvgColor  "<<timer.seconds()*1000<<"ms"<<std::endl;
#endif

#ifdef REPORT

	timer.start();
#endif
	//GraphCutOptimize(m_spPtr,m_nPixel,avgE,2,m_width,m_height,m_result);
	//MaxFlowOptimize(m_spPtr,m_nPixel,avgE,2,m_width,m_height,m_result);
	GridCutOptimize(m_spPtr,m_nPixel,avgE,2,m_width,m_height,m_result);
#ifdef REPORT
	timer.stop();
	std::cout<<"GraphCutOptimize  "<<timer.seconds()*1000<<"ms"<<std::endl;
#endif

#ifdef REPORT	
	timer.start();
#endif
	
	resultImg = cv::Scalar(0);
	unsigned char* imgPtr = resultImg.data;
	for(int i=0; i<m_nPixel; i++)
	{
		if(m_result[i] == 1)			
		{
			/*for(int j=0; j<m_spPtr[i].pixels.size(); j++)
			{
			int idx = m_spPtr[i].pixels[j].first + m_spPtr[i].pixels[j].second*m_width;
			imgPtr[idx] = 0xff;
			}*/
			int k = m_centers[i].xy.x;
			int j = m_centers[i].xy.y;
			int radius = m_step+5;
			for (int x = k- radius; x<= k+radius; x++)
			{
				for(int y = j - radius; y<= j+radius; y++)
				{
					if  (x<0 || x>m_width-1 || y<0 || y> m_height-1)
						continue;
					int idx = x+y*m_width;
					if (m_labels[idx] == i)
					{					
						imgPtr[idx] = 0xff;
					}					
				}
			}
		}	
	}
	
#ifdef REPORT
	timer.stop();
	std::cout<<"imwrite  "<<timer.seconds()*1000<<"ms"<<std::endl;
#endif
	

#ifdef REPORT
	timer0.stop();
	std::cout<<"optimize one frame  "<<timer0.seconds()*1000<<"ms"<<std::endl;
#endif

}
void MRFOptimize::Optimize(GpuSuperpixel* GS,uchar4 * d_rgba,cv::Mat& maskImg, cv::Mat& featureMaskImg, cv::Mat& resultImg)
{
	#ifdef REPORT
	nih::Timer timer;
	nih::Timer timer0;
	timer0.start();
	timer.start();
#endif
	
	//superpixel			

	const unsigned char* maskImgData = maskImg.data;
	const unsigned char* featureMaskData = featureMaskImg.data;
#ifdef REPORT
	timer.stop();
	std::cout<<"read image  "<<timer.seconds()*1000 <<"ms"<<std::endl;
#endif

	int numlabels(0);

#ifdef REPORT
	GpuTimer gtimer;
	gtimer.Start();
#endif
	GS->DSuperpixel(d_rgba,numlabels,m_labels,m_centers);
	
	//cv::Mat labelImg(m_height,m_width,CV_8U);
	//labelImg = cv::Scalar(0);
	//uchar* ptr = labelImg.data;
	//std::ofstream labelF("label.tmp");
	//for(int y=0; y<m_height; y++)
	//{
	//	for(int x=0; x<m_width; x++)
	//	{
	//		int idx = x+ y*m_width;
	//		labelF<<m_labels[idx]<<" ";			
	//	}
	//	labelF<<"\n ";
	//}
	//labelF.close();
	//cv::imwrite("label.jpg",labelImg);
	//ComSuperpixel cs;
	//cs.Superpixel(m_idata,m_width,m_height,5,0.9,numlabels,m_labels);
#ifdef REPORT
	gtimer.Stop();
	std::cout<<"GPU SuperPixel "<<gtimer.Elapsed()<<"ms"<<std::endl;
#endif
	/*std::ofstream fileout("out.tmp.");
	for(int i=0; i<width; i++)
	{
	for(int j=0; j<height; j++)
	{
	fileout<<labels[i+j*width]<<"\t ";
	}
	fileout<<"\n";
	}
	fileout.close();*/

	/*SLIC aslic;
	cudaMemcpy(m_idata,d_rgba,sizeof(uchar4)*m_width*m_height,cudaMemcpyDeviceToHost);
	aslic.DrawContoursAroundSegments(m_idata, m_labels, m_width, m_height,0x00ff00);
	PictureHandler handler;
	handler.SavePicture(m_idata,m_width,m_height,std::string("mysuper.jpg"),std::string(".\\"));*/
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
//#ifdef REPORT
//	timer.start();
//#endif

	/*size_t spSize(0);
	GetSegment2DArray(m_spPtr,spSize,m_labels,m_width,m_height);
	cv::Mat  sp(m_height,m_width,CV_8UC3);
	for(int i=0; i<spSize; i++)
	{
		cv::Vec3b color(rand()%255,rand()%255,rand()%255);
		for(int j=0; j<m_spPtr[i].pixels.size(); j++)
		{

			sp.at<cv::Vec3b>(m_spPtr[i].pixels[j].second,m_spPtr[i].pixels[j].first) = color;
		}
	}	
	cv::imwrite("sp.jpg",sp);*/
	
//#ifdef REPORT
//	timer.stop();
//	std::cout<<"GetSegment2DArray  "<<timer.seconds()*1000<<"ms"<<std::endl;
//#endif
//
#ifdef REPORT

	timer.start();
#endif
	//ComputeAvgColor(m_spPtr,spSize,m_width,m_height,m_idata,maskImgData);
	GetSuperpixels(maskImgData,featureMaskData);
	//ProbImage(m_spPtr,m_labels,m_nPixel,m_width,m_height);
	//std::ofstream outFile("nb.tmp");
	//for(int id = 0; id<m_nPixel; id++)
	//{
	//	outFile<<id<<" ";
	//	for(int i=0; i<m_spPtr[id].neighbors.size();i++)
	//	{
	//		outFile<<m_spPtr[id].neighbors[i]->lable<<" ";
	//	}
	//	outFile<<std::endl;
	//}
	//outFile.close();
	//NeighborsImage(m_spPtr,m_labels,m_nPixel,m_width,m_height);
#ifdef REPORT
	timer.stop();
	std::cout<<"GetSuperpixels  "<<timer.seconds()*1000<<"ms"<<std::endl;
#endif
#ifdef REPORT
	timer.start();
#endif
	float avgE = 0;
	size_t count = 0;	
	for(int i=0; i<m_nPixel; i++)
	{
		for (int j=0; j< m_neighbor[i].size(); j++)
		{
			if (m_centers[i].nPoints > 0 && m_centers[m_neighbor[i][j]].nPoints >0 )
			{				
				avgE += abs(m_spPtr[i].avgColor-m_spPtr[m_neighbor[i][j]].avgColor);
				count++;
			}
		}
		//for (int j=0; j< m_spPtr[i].neighbors.size(); j++)
		//{
		//	if (m_spPtr[i].lable < m_spPtr[i].neighbors[j]->lable)
		//	{

		//		avgE += abs(m_spPtr[i].avgColor-m_spPtr[i].neighbors[j]->avgColor);
		//		count++;
		//	}

		//}
	}
	avgE /= count;
	avgE = 1/(2*avgE);
	//std::cout<<"avg e "<<avgE<<std::endl;
#ifdef REPORT
	timer.stop();
	std::cout<<"ComputeAvgColor  "<<timer.seconds()*1000<<"ms"<<std::endl;
#endif

#ifdef REPORT

	timer.start();
#endif
	//GraphCutOptimize(m_spPtr,m_nPixel,avgE,2,m_width,m_height,m_result);
	//MaxFlowOptimize(m_spPtr,m_nPixel,avgE,2,m_width,m_height,m_result);
	GridCutOptimize(m_spPtr,m_nPixel,avgE,2,m_width,m_height,m_result);
#ifdef REPORT
	timer.stop();
	std::cout<<"GraphCutOptimize  "<<timer.seconds()*1000<<"ms"<<std::endl;
#endif

#ifdef REPORT	
	timer.start();
#endif
	
	resultImg = cv::Scalar(0);
	unsigned char* imgPtr = resultImg.data;
	for(int i=0; i<m_nPixel; i++)
	{
		if(m_result[i] == 1)			
		{
			/*for(int j=0; j<m_spPtr[i].pixels.size(); j++)
			{
			int idx = m_spPtr[i].pixels[j].first + m_spPtr[i].pixels[j].second*m_width;
			imgPtr[idx] = 0xff;
			}*/
			int k = m_centers[i].xy.x;
			int j = m_centers[i].xy.y;
			int radius = m_step+5;
			for (int x = k- radius; x<= k+radius; x++)
			{
				for(int y = j - radius; y<= j+radius; y++)
				{
					if  (x<0 || x>m_width-1 || y<0 || y> m_height-1)
						continue;
					int idx = x+y*m_width;
					if (m_labels[idx] == i)
					{					
						imgPtr[idx] = 0xff;
					}					
				}
			}
		}	
	}
	
#ifdef REPORT
	timer.stop();
	std::cout<<"imwrite  "<<timer.seconds()*1000<<"ms"<<std::endl;
#endif
	//delete[] m_spPtr;

#ifdef REPORT
	timer0.stop();
	std::cout<<"optimize one frame  "<<timer0.seconds()*1000<<"ms"<<std::endl;
#endif

}
void MRFOptimize::ComuteSuperpixel(GpuSuperpixel* GS, uchar4* d_rgba)
{
	int numlabels;
	GS->DSuperpixel(d_rgba,numlabels,m_labels,m_centers);
	//计算超像素之间的平均颜色差
	m_avgE= 0;
	size_t count = 0;	
	for(int i=0; i<m_nPixel; i++)
	{
		for (int j=0; j< m_neighbor[i].size(); j++)
		{
			if (m_centers[i].nPoints > 0 && m_centers[m_neighbor[i][j]].nPoints >0 )
			{				
				m_avgE += abs(m_spPtr[i].avgColor-m_spPtr[m_neighbor[i][j]].avgColor);
				count++;
			}
		}		
	}
	m_avgE /= count;
	//std::cout<<"avg e "<<m_avgE<<std::endl;
}
void MRFOptimize::Optimize(const cv::Mat& maskImg, const cv::Mat& featureImg, cv::Mat& resultImg)
{
#ifdef REPORT
	nih::Timer timer;
	nih::Timer timer0;
	timer0.start();
	timer.start();
#endif
	
	//superpixel			

	const unsigned char* maskImgData = maskImg.data;
	const unsigned char* featureMaskData = featureImg.data;
#ifdef REPORT
	timer.stop();
	std::cout<<"read image  "<<timer.seconds()*1000 <<"ms"<<std::endl;
#endif

	int numlabels(0);

#ifdef REPORT
	GpuTimer gtimer;
	gtimer.Start();
#endif	
	

#ifdef REPORT
	gtimer.Stop();
	std::cout<<"GPU SuperPixel "<<gtimer.Elapsed()<<"ms"<<std::endl;
#endif

#ifdef REPORT

	timer.start();
#endif
	//ComputeAvgColor(m_spPtr,spSize,m_width,m_height,m_idata,maskImgData);
	GetSuperpixels(maskImgData,featureMaskData);
	
#ifdef REPORT
	timer.stop();
	std::cout<<"GetSuperpixels  "<<timer.seconds()*1000<<"ms"<<std::endl;
#endif
#ifdef REPORT
	timer.start();
#endif
	float avgE = 0;
	size_t count = 0;	
	for(int i=0; i<m_nPixel; i++)
	{
		for (int j=0; j< m_neighbor[i].size(); j++)
		{
			if (m_centers[i].nPoints > 0 && m_centers[m_neighbor[i][j]].nPoints >0 )
			{				
				avgE += abs(m_spPtr[i].avgColor-m_spPtr[m_neighbor[i][j]].avgColor);
				count++;
			}
		}		
	}
	avgE /= count;
	avgE = 1/(2*avgE);
	//std::cout<<"avg e "<<avgE<<std::endl;
#ifdef REPORT
	timer.stop();
	std::cout<<"ComputeAvgColor  "<<timer.seconds()*1000<<"ms"<<std::endl;
#endif

#ifdef REPORT

	timer.start();
#endif
	//GraphCutOptimize(m_spPtr,m_nPixel,avgE,2,m_width,m_height,m_result);
	//MaxFlowOptimize(m_spPtr,m_nPixel,avgE,2,m_width,m_height,m_result);
	GridCutOptimize(m_spPtr,m_nPixel,avgE,2,m_width,m_height,m_result);
#ifdef REPORT
	timer.stop();
	std::cout<<"GraphCutOptimize  "<<timer.seconds()*1000<<"ms"<<std::endl;
#endif

#ifdef REPORT	
	timer.start();
#endif
	
	resultImg = cv::Scalar(0);
	unsigned char* imgPtr = resultImg.data;
	for(int i=0; i<m_nPixel; i++)
	{
		if(m_result[i] == 1)			
		{
			/*for(int j=0; j<m_spPtr[i].pixels.size(); j++)
			{
			int idx = m_spPtr[i].pixels[j].first + m_spPtr[i].pixels[j].second*m_width;
			imgPtr[idx] = 0xff;
			}*/
			int k = m_centers[i].xy.x;
			int j = m_centers[i].xy.y;
			int radius = m_step+5;
			for (int x = k- radius; x<= k+radius; x++)
			{
				for(int y = j - radius; y<= j+radius; y++)
				{
					if  (x<0 || x>m_width-1 || y<0 || y> m_height-1)
						continue;
					int idx = x+y*m_width;
					if (m_labels[idx] == i)
					{					
						imgPtr[idx] = 0xff;
					}					
				}
			}
		}	
	}
	
#ifdef REPORT
	timer.stop();
	std::cout<<"imwrite  "<<timer.seconds()*1000<<"ms"<<std::endl;
#endif
	//delete[] m_spPtr;

#ifdef REPORT
	timer0.stop();
	std::cout<<"optimize one frame  "<<timer0.seconds()*1000<<"ms"<<std::endl;
#endif

}
void MRFOptimize::Optimize(GpuSuperpixel* GS, const cv::Mat& origImg, const cv::Mat& maskImg,const cv::Mat& lastMaskImg, const cv::Mat& flow, const cv::Mat& homography,cv::Mat& resultImg)
{
	//superpixel		
	cv::Mat FImg(origImg.size(), CV_8UC4, m_imgData);
	cv::Mat continuousRGBA(origImg.size(), CV_8UC4, m_idata);
	cv::cvtColor(origImg,continuousRGBA,CV_BGR2BGRA);
	continuousRGBA.convertTo(FImg,CV_8UC4);

	int numlabels(0);
	GS->Superpixel(m_imgData,numlabels,m_labels,m_centers);

	cv::Mat lMask = cv::Mat(m_height,m_width,CV_8U);
	lMask = cv::Scalar(0);
	const uchar* maskPtr = maskImg.data;
	const uchar* prevMaskPtr = lMask.data;
	uchar* lastMaskPtr = lastMaskImg.data;
	
	
	//mask from last mask
	for(int i=0; i< m_width; i++)
	{
		for(int j=0; j<m_height; j++)
		{
			int idx = i + j*m_width;
			int idx_flt32 = idx*4*2;
			if (lastMaskPtr[idx] == 0xff)
			{
				
			
				float* flowPtr = (float*)(flow.data+ idx_flt32);
				float dx = flowPtr[0];
				float dy = flowPtr[1];
				//std::cout<<dx<<" , "<<dy<<std::endl;
				int wx = (i+dx);
				int wy = (j+dy);
				wx =  wx <0 ? 0 :wx;
				wx = wx > m_width-1 ? m_width-1 : wx;
				wy = wy<0 ? 0 : wy;
				wy = wy > m_height-1 ? m_height-1 : wy;
				int tid = wx + wy*m_width;
				lMask.data[tid] = 0xff;
			}
		}
	}
	resultImg = lMask;
	
	
	
	GetSuperpixels(maskPtr,prevMaskPtr,flow,homography);
	
	float avgE = 0;
	size_t count = 0;	
	for(int i=0; i<m_nPixel; i++)
	{
		for (int j=0; j< m_neighbor[i].size(); j++)
		{
			if (m_centers[i].nPoints > 0 && m_centers[m_neighbor[i][j]].nPoints >0 )
			{				
				avgE += abs(m_spPtr[i].avgColor-m_spPtr[m_neighbor[i][j]].avgColor);
				count++;
			}
		}
		
	}
	avgE /= count;
	avgE = 1/(2*avgE);
	//std::cout<<"avg e "<<avgE<<std::endl;
#ifdef REPORT
	timer.stop();
	std::cout<<"ComputeAvgColor  "<<timer.seconds()*1000<<"ms"<<std::endl;
#endif

#ifdef REPORT

	timer.start();
#endif
	//GraphCutOptimize(m_spPtr,m_nPixel,avgE,2,m_width,m_height,m_result);
	MaxFlowOptimize(m_spPtr,m_nPixel,avgE,2,m_width,m_height,m_result);
	//GridCutOptimize(m_spPtr,m_nPixel,avgE,2,m_width,m_height,m_result);
#ifdef REPORT
	timer.stop();
	std::cout<<"GraphCutOptimize  "<<timer.seconds()*1000<<"ms"<<std::endl;
#endif

#ifdef REPORT	
	timer.start();
#endif
	resultImg = cv::Mat(m_height,m_width,CV_8U);
	resultImg = cv::Scalar(0);
	unsigned char* imgPtr = resultImg.data;
	for(int i=0; i<m_nPixel; i++)
	{
		if(m_result[i] == 1)			
		{
			/*for(int j=0; j<m_spPtr[i].pixels.size(); j++)
			{
			int idx = m_spPtr[i].pixels[j].first + m_spPtr[i].pixels[j].second*m_width;
			imgPtr[idx] = 0xff;
			}*/
			int k = m_centers[i].xy.x;
			int j = m_centers[i].xy.y;
			int radius = m_step+5;
			for (int x = k- radius; x<= k+radius; x++)
			{
				for(int y = j - radius; y<= j+radius; y++)
				{
					if  (x<0 || x>m_width-1 || y<0 || y> m_height-1)
						continue;
					int idx = x+y*m_width;
					if (m_labels[idx] == i)
					{					
						imgPtr[idx] = 0xff;
					}					
				}
			}
		}	
	}
}
void MRFOptimize::Optimize(GpuSuperpixel* GS,uchar4 * d_rgba,cv::Mat& maskImg, cv::Mat& featureMaskImg, float* distance,cv::Mat& resultImg)
{
	#ifdef REPORT
	nih::Timer timer;
	nih::Timer timer0;
	timer0.start();
	timer.start();
#endif
	
	//superpixel			

	const unsigned char* maskImgData = maskImg.data;
	const unsigned char* featureMaskData = featureMaskImg.data;
	
#ifdef REPORT
	timer.stop();
	std::cout<<"read image  "<<timer.seconds()*1000 <<"ms"<<std::endl;
#endif

	int numlabels(0);

#ifdef REPORT
	GpuTimer gtimer;
	gtimer.Start();
#endif
	GS->DSuperpixel(d_rgba,numlabels,m_labels,m_centers);
	
	//cv::Mat labelImg(m_height,m_width,CV_8U);
	//labelImg = cv::Scalar(0);
	//uchar* ptr = labelImg.data;
	//std::ofstream labelF("label.tmp");
	//for(int y=0; y<m_height; y++)
	//{
	//	for(int x=0; x<m_width; x++)
	//	{
	//		int idx = x+ y*m_width;
	//		labelF<<m_labels[idx]<<" ";			
	//	}
	//	labelF<<"\n ";
	//}
	//labelF.close();
	//cv::imwrite("label.jpg",labelImg);
	//ComSuperpixel cs;
	//cs.Superpixel(m_idata,m_width,m_height,5,0.9,numlabels,m_labels);
#ifdef REPORT
	gtimer.Stop();
	std::cout<<"GPU SuperPixel "<<gtimer.Elapsed()<<"ms"<<std::endl;
#endif
	/*std::ofstream fileout("out.tmp.");
	for(int i=0; i<width; i++)
	{
	for(int j=0; j<height; j++)
	{
	fileout<<labels[i+j*width]<<"\t ";
	}
	fileout<<"\n";
	}
	fileout.close();*/

	/*SLIC aslic;
	cudaMemcpy(m_idata,d_rgba,sizeof(uchar4)*m_width*m_height,cudaMemcpyDeviceToHost);
	aslic.DrawContoursAroundSegments(m_idata, m_labels, m_width, m_height,0x00ff00);
	PictureHandler handler;
	handler.SavePicture(m_idata,m_width,m_height,std::string("mysuper.jpg"),std::string(".\\"));*/
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
//#ifdef REPORT
//	timer.start();
//#endif

	/*size_t spSize(0);
	GetSegment2DArray(m_spPtr,spSize,m_labels,m_width,m_height);
	cv::Mat  sp(m_height,m_width,CV_8UC3);
	for(int i=0; i<spSize; i++)
	{
		cv::Vec3b color(rand()%255,rand()%255,rand()%255);
		for(int j=0; j<m_spPtr[i].pixels.size(); j++)
		{

			sp.at<cv::Vec3b>(m_spPtr[i].pixels[j].second,m_spPtr[i].pixels[j].first) = color;
		}
	}	
	cv::imwrite("sp.jpg",sp);*/
	
//#ifdef REPORT
//	timer.stop();
//	std::cout<<"GetSegment2DArray  "<<timer.seconds()*1000<<"ms"<<std::endl;
//#endif
//
#ifdef REPORT

	timer.start();
#endif
	//ComputeAvgColor(m_spPtr,spSize,m_width,m_height,m_idata,maskImgData);
	GetSuperpixels(maskImgData,featureMaskData,distance);
	//ProbImage(m_spPtr,m_labels,m_nPixel,m_width,m_height);
	//std::ofstream outFile("nb.tmp");
	//for(int id = 0; id<m_nPixel; id++)
	//{
	//	outFile<<id<<" ";
	//	for(int i=0; i<m_spPtr[id].neighbors.size();i++)
	//	{
	//		outFile<<m_spPtr[id].neighbors[i]->lable<<" ";
	//	}
	//	outFile<<std::endl;
	//}
	//outFile.close();
	//NeighborsImage(m_spPtr,m_labels,m_nPixel,m_width,m_height);
#ifdef REPORT
	timer.stop();
	std::cout<<"GetSuperpixels  "<<timer.seconds()*1000<<"ms"<<std::endl;
#endif
#ifdef REPORT
	timer.start();
#endif
	float avgE = 0;
	size_t count = 0;	
	for(int i=0; i<m_nPixel; i++)
	{
		for (int j=0; j< m_neighbor[i].size(); j++)
		{
			if (m_centers[i].nPoints > 0 && m_centers[m_neighbor[i][j]].nPoints >0 )
			{				
				avgE += abs(m_spPtr[i].avgColor-m_spPtr[m_neighbor[i][j]].avgColor);
				count++;
			}
		}
		//for (int j=0; j< m_spPtr[i].neighbors.size(); j++)
		//{
		//	if (m_spPtr[i].lable < m_spPtr[i].neighbors[j]->lable)
		//	{

		//		avgE += abs(m_spPtr[i].avgColor-m_spPtr[i].neighbors[j]->avgColor);
		//		count++;
		//	}

		//}
	}
	avgE /= count;
	avgE = 1/(2*avgE);
	//std::cout<<"avg e "<<avgE<<std::endl;
#ifdef REPORT
	timer.stop();
	std::cout<<"ComputeAvgColor  "<<timer.seconds()*1000<<"ms"<<std::endl;
#endif

#ifdef REPORT

	timer.start();
#endif
	//GraphCutOptimize(m_spPtr,m_nPixel,avgE,2,m_width,m_height,m_result);
	//MaxFlowOptimize(m_spPtr,m_nPixel,avgE,2,m_width,m_height,m_result);
	GridCutOptimize(m_spPtr,m_nPixel,avgE,2,m_width,m_height,m_result);
#ifdef REPORT
	timer.stop();
	std::cout<<"GraphCutOptimize  "<<timer.seconds()*1000<<"ms"<<std::endl;
#endif

#ifdef REPORT	
	timer.start();
#endif
	
	resultImg = cv::Scalar(0);
	unsigned char* imgPtr = resultImg.data;
	for(int i=0; i<m_nPixel; i++)
	{
		if(m_result[i] == 1)			
		{
			/*for(int j=0; j<m_spPtr[i].pixels.size(); j++)
			{
			int idx = m_spPtr[i].pixels[j].first + m_spPtr[i].pixels[j].second*m_width;
			imgPtr[idx] = 0xff;
			}*/
			int k = m_centers[i].xy.x;
			int j = m_centers[i].xy.y;
			int radius = m_step+5;
			for (int x = k- radius; x<= k+radius; x++)
			{
				for(int y = j - radius; y<= j+radius; y++)
				{
					if  (x<0 || x>m_width-1 || y<0 || y> m_height-1)
						continue;
					int idx = x+y*m_width;
					if (m_labels[idx] == i)
					{					
						imgPtr[idx] = 0xff;
					}					
				}
			}
		}	
	}
	
#ifdef REPORT
	timer.stop();
	std::cout<<"imwrite  "<<timer.seconds()*1000<<"ms"<<std::endl;
#endif
	//delete[] m_spPtr;

#ifdef REPORT
	timer0.stop();
	std::cout<<"optimize one frame  "<<timer0.seconds()*1000<<"ms"<<std::endl;
#endif

}
void MRFOptimize::Optimize(GpuSuperpixel* GS, const string& originalImgName, const string& maskImgName,  const string& featuremaskImgName,const string& resultImgName)
{
	#ifdef REPORT
	nih::Timer timer;
	nih::Timer timer0;
	timer0.start();
	timer.start();
#endif
	/*m_spPtr = new SuperPixel[m_width*m_height];*/
	//superpixel
	cv::Mat img = cv::imread(originalImgName);		
	cv::Mat continuousRGBA(img.size(), CV_8UC4, m_idata);	
	cv::Mat FImg(img.size(), CV_8UC4, m_imgData);
	cv::cvtColor(img,continuousRGBA,CV_BGR2BGRA,4);
	continuousRGBA.convertTo(FImg,CV_8UC4);


	cv::Mat maskImg = cv::imread(maskImgName);
	cv::cvtColor(maskImg,maskImg,CV_BGR2GRAY);
	maskImg.convertTo(maskImg,CV_8U);

	cv::Mat featureMaskImg = cv::imread(featuremaskImgName);
	cv::cvtColor(featureMaskImg,featureMaskImg,CV_BGR2GRAY);
	featureMaskImg.convertTo(featureMaskImg,CV_8U);
	
	const unsigned char* maskImgData = maskImg.data;
	const unsigned char* featureMaskData = featureMaskImg.data;
#ifdef REPORT
	timer.stop();
	std::cout<<"read image  "<<timer.seconds()*1000 <<"ms"<<std::endl;
#endif

	int numlabels(0);

#ifdef REPORT
	GpuTimer gtimer;
	gtimer.Start();
#endif
	GS->Superpixel(m_imgData,numlabels,m_labels,m_centers);
	
	//cv::Mat labelImg(m_height,m_width,CV_8U);
	//labelImg = cv::Scalar(0);
	//uchar* ptr = labelImg.data;
	//std::ofstream labelF("label.tmp");
	//for(int y=0; y<m_height; y++)
	//{
	//	for(int x=0; x<m_width; x++)
	//	{
	//		int idx = x+ y*m_width;
	//		labelF<<m_labels[idx]<<" ";			
	//	}
	//	labelF<<"\n ";
	//}
	//labelF.close();
	//cv::imwrite("label.jpg",labelImg);
	//ComSuperpixel cs;
	//cs.Superpixel(m_idata,m_width,m_height,5,0.9,numlabels,m_labels);
#ifdef REPORT
	gtimer.Stop();
	std::cout<<"GPU SuperPixel "<<gtimer.Elapsed()<<"ms"<<std::endl;
#endif
	/*std::ofstream fileout("out.tmp.");
	for(int i=0; i<width; i++)
	{
	for(int j=0; j<height; j++)
	{
	fileout<<labels[i+j*width]<<"\t ";
	}
	fileout<<"\n";
	}
	fileout.close();*/

	/*SLIC aslic;
	aslic.DrawContoursAroundSegments(m_idata, m_labels, m_width, m_height,0x00ff00);
	PictureHandler handler;
	handler.SavePicture(m_idata,m_width,m_height,std::string("mysuper.jpg"),std::string(".\\"));*/
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
//#ifdef REPORT
//	timer.start();
//#endif

	/*size_t spSize(0);
	GetSegment2DArray(m_spPtr,spSize,m_labels,m_width,m_height);
	cv::Mat  sp(m_height,m_width,CV_8UC3);
	for(int i=0; i<spSize; i++)
	{
		cv::Vec3b color(rand()%255,rand()%255,rand()%255);
		for(int j=0; j<m_spPtr[i].pixels.size(); j++)
		{

			sp.at<cv::Vec3b>(m_spPtr[i].pixels[j].second,m_spPtr[i].pixels[j].first) = color;
		}
	}	
	cv::imwrite("sp.jpg",sp);*/
	
//#ifdef REPORT
//	timer.stop();
//	std::cout<<"GetSegment2DArray  "<<timer.seconds()*1000<<"ms"<<std::endl;
//#endif
//
#ifdef REPORT

	timer.start();
#endif
	//ComputeAvgColor(m_spPtr,spSize,m_width,m_height,m_idata,maskImgData);
	GetSuperpixels(maskImgData,featureMaskData);
	//ProbImage(m_spPtr,m_labels,m_nPixel,m_width,m_height);
	//std::ofstream outFile("nb.tmp");
	//for(int id = 0; id<m_nPixel; id++)
	//{
	//	outFile<<id<<" ";
	//	for(int i=0; i<m_spPtr[id].neighbors.size();i++)
	//	{
	//		outFile<<m_spPtr[id].neighbors[i]->lable<<" ";
	//	}
	//	outFile<<std::endl;
	//}
	//outFile.close();
	//NeighborsImage(m_spPtr,m_labels,m_nPixel,m_width,m_height);
#ifdef REPORT
	timer.stop();
	std::cout<<"GetSuperpixels  "<<timer.seconds()*1000<<"ms"<<std::endl;
#endif
#ifdef REPORT
	timer.start();
#endif
	float avgE = 0;
	size_t count = 0;	
	for(int i=0; i<m_nPixel; i++)
	{
		for (int j=0; j< m_neighbor[i].size(); j++)
		{
			if (m_centers[i].nPoints > 0 && m_centers[m_neighbor[i][j]].nPoints >0 )
			{				
				avgE += abs(m_spPtr[i].avgColor-m_spPtr[m_neighbor[i][j]].avgColor);
				count++;
			}
		}
		//for (int j=0; j< m_spPtr[i].neighbors.size(); j++)
		//{
		//	if (m_spPtr[i].lable < m_spPtr[i].neighbors[j]->lable)
		//	{

		//		avgE += abs(m_spPtr[i].avgColor-m_spPtr[i].neighbors[j]->avgColor);
		//		count++;
		//	}

		//}
	}
	avgE /= count;
	avgE = 1/(2*avgE);
	//std::cout<<"avg e "<<avgE<<std::endl;
#ifdef REPORT
	timer.stop();
	std::cout<<"ComputeAvgColor  "<<timer.seconds()*1000<<"ms"<<std::endl;
#endif

#ifdef REPORT

	timer.start();
#endif
	//GraphCutOptimize(m_spPtr,m_nPixel,avgE,2,m_width,m_height,m_result);
	MaxFlowOptimize(m_spPtr,m_nPixel,avgE,2,m_width,m_height,m_result);
	//GridCutOptimize(m_spPtr,m_nPixel,avgE,2,m_width,m_height,m_result);
#ifdef REPORT
	timer.stop();
	std::cout<<"GraphCutOptimize  "<<timer.seconds()*1000<<"ms"<<std::endl;
#endif

#ifdef REPORT	
	timer.start();
#endif
	cv::Mat rimg(m_height,m_width,CV_8U);
	rimg = cv::Scalar(0);
	unsigned char* imgPtr = rimg.data;
	for(int i=0; i<m_nPixel; i++)
	{
		if(m_result[i] == 1)			
		{
			/*for(int j=0; j<m_spPtr[i].pixels.size(); j++)
			{
			int idx = m_spPtr[i].pixels[j].first + m_spPtr[i].pixels[j].second*m_width;
			imgPtr[idx] = 0xff;
			}*/
			int k = m_centers[i].xy.x;
			int j = m_centers[i].xy.y;
			int radius = m_step+5;
			for (int x = k- radius; x<= k+radius; x++)
			{
				for(int y = j - radius; y<= j+radius; y++)
				{
					if  (x<0 || x>m_width-1 || y<0 || y> m_height-1)
						continue;
					int idx = x+y*m_width;
					if (m_labels[idx] == i)
					{					
						imgPtr[idx] = 0xff;
					}					
				}
			}
		}	
	}
	cv::imwrite(resultImgName,rimg);
#ifdef REPORT
	timer.stop();
	std::cout<<"imwrite  "<<timer.seconds()*1000<<"ms"<<std::endl;
#endif
	//delete[] m_spPtr;

#ifdef REPORT
	timer0.stop();
	std::cout<<"optimize one frame  "<<timer0.seconds()*1000<<"ms"<<std::endl;
#endif

}
bool isNeighbour(int label, int x, int y, int width, int height, int* labels)
{ 
	const int dx8[8] = {-1, -1,  0,  1, 1, 1, 0, -1};
	const int dy8[8] = { 0, -1, -1, -1, 0, 1, 1,  1};
	int index = x + y*width;
	if ( labels[index] != label)
	{
		for(int i=0; i<8; i++)
		{
			int k = x + dx8[i];
			int j = y + dy8[i];
			if ( k<0 )
				k=0;
			if(k>=width)
				k=width-1;
			if(j<0)
				j=0;
			if(j>=height)
				j=height-1;
			if (labels[ k+ j*width] == label)
				return true;
		}
		return false;
	}
	else
		return false;
}
void MRFOptimize::GetSuperpixels(const unsigned char* mask)
{
	
	int size = m_width*m_height;
	for(int i=0; i<m_nPixel; i++)
	{		
		int k = m_centers[i].xy.x;
		int j = m_centers[i].xy.y;
		if (m_centers[i].nPoints ==0)			
		{
			m_spPtr[i].ps = 0;
			m_spPtr[i].avgColor = 0;
			m_spPtr[i].idx = i;
			m_spPtr[i].lable = i;
		}
		else
		{
			m_spPtr[i].avgColor = (m_centers[i].rgb.x+m_centers[i].rgb.y + m_centers[i].rgb.z)/3;
			m_spPtr[i].idx = i;
			m_spPtr[i].lable = i;

			float n = 0;
			//以原来的中心点为中心，step +2　为半径进行更新
			int radius = m_step;
			for (int x = k- radius; x<= k+radius; x++)
			{
				for(int y = j - radius; y<= j+radius; y++)
				{
					if  (x<0 || x>m_width-1 || y<0 || y> m_height-1)
						continue;
					int idx = x+y*m_width;
					//std::cout<<idx<<std::endl;
					if (m_labels[idx] == i )
					{		
						
						if ( mask[idx] == 0xff)
							n++;
					}
					//else if(isNeighbour(i,x,y,m_width,m_height,m_labels))
					//{
					//	if (find(m_spPtr[i].neighbors.begin(),m_spPtr[i].neighbors.end(),&m_spPtr[m_labels[idx]]) == m_spPtr[i].neighbors.end())
					//		m_spPtr[i].neighbors.push_back(&m_spPtr[m_labels[idx]]);
					//}
				}
			}
			m_spPtr[i].ps = n/m_centers[i].nPoints;
		}
	}
	/*ProbImage(m_spPtr,m_labels,m_nPixel,m_width,m_height);
	NeighborsImage(m_spPtr,m_labels,m_nPixel,m_width,m_height,m_neighbor);*/
}
void DistanceMat(const cv::Mat& homography, const cv::Mat& flow, cv::Mat& dist)
{
	int width = flow.cols;
	int height = flow.rows;
	double * homoPtr = (double*)homography.data;
	float * flowPtr = (float*)flow.data;
	float * distPtr = (float*)dist.data;
	float distMin(1e5);
	float distMax(0);
	float distAvg(0);
	for( int i = 0; i<width; i++)
	{
		for(int j=0; j<height; j++)
		{
			
			float wx = homoPtr[0]*i + homoPtr[1]*j + homoPtr[2];
			float wy = homoPtr[3]*i + homoPtr[4]*j + homoPtr[5];
			float w = homoPtr[6]*i + homoPtr[7]*j + homoPtr[8];
			wx /= w;
			wy /= w;
			int idx = (i + j*width);
			float fx = i+flowPtr[idx*2];
			float fy = j+flowPtr[idx*2+1];
			float dx = fx - wx;
			float dy = fy - wy;
			float d = dx*dx + dy*dy;
			distPtr[idx] = d;
			distAvg += d;
			if (d>distMax)
				distMax = d;
			if (d<distMin)
				distMin = d;
		}
	}
	distAvg/=(width*height);
	std::cout<<distAvg<<std::endl;
	cv::Mat mask;
	dist.convertTo(mask,CV_8U,255/(distMax-distMin),0);
	cv::imwrite("distMask.jpg",mask);
}
void MRFOptimize::GetSuperpixels(const unsigned char* mask, const uchar* lastMask, const cv::Mat& flow, const cv::Mat& homography)
{
	std::vector<double> histogram(256);
	memset(&histogram[0],0,sizeof(double)*256);
	cv::Mat dist(m_height,m_width,CV_32F);
	DistanceMat(homography,flow,dist);
	int size = m_width*m_height;
	float avgDis =0 ;
	double* data = (double*)homography.data;
	float* flowPtr = (float*)flow.data;
	float* distPtr = (float*)dist.data;
	for(int i=0; i<m_nPixel; i++)
	{		
		int k = m_centers[i].xy.x;
		int j = m_centers[i].xy.y;
		if (m_centers[i].nPoints ==0)			
		{
			m_spPtr[i].ps = 0;
			m_spPtr[i].avgColor = 0;
			m_spPtr[i].idx = i;
			m_spPtr[i].lable = i;
			m_spPtr[i].distance = 0;
		}
		else
		{
			m_spPtr[i].avgColor = (m_centers[i].rgb.x+m_centers[i].rgb.y + m_centers[i].rgb.z)/3;
			m_spPtr[i].idx = i;
			m_spPtr[i].lable = i;

			float n = 0;			
			float d(0);
			int c(0);
			//以原来的中心点为中心，step +2　为半径进行更新
			int radius = m_step;
			for (int x = k- radius; x<= k+radius; x++)
			{
				for(int y = j - radius; y<= j+radius; y++)
				{
					if  (x<0 || x>m_width-1 || y<0 || y> m_height-1)
						continue;
					int idx = x+y*m_width;
					int flowIdx = idx*8;
					//std::cout<<idx<<std::endl;
					if (m_labels[idx] == i )
					{		
						c++;
						d+= distPtr[idx];
						if ( mask[idx] == 0xff /*|| lastMask[idx] == 0xf*/)
						{
							
								n++;
						}
							
					}					
				}
			}
			m_spPtr[i].ps  = n/c;
			m_spPtr[i].distance = d/c;
			histogram[(uchar)m_spPtr[i].avgColor] += m_spPtr[i].ps;
		}
	}
	double minV(1e10),maxV(0);
	int minIdx,maxIdx;
	for(int i=0; i<histogram.size(); i++)
	{
		if (histogram[i]>maxV)
		{
			maxV = histogram[i];
			maxIdx = i;
		}
		if (histogram[i] < minV)
		{
			minV = histogram[i];
			minIdx = i;
		}
	}
	cv::Mat grayMask(m_height,m_width,CV_8U);
	grayMask = cv::Scalar(0);
	for(int i=0; i<m_nPixel; i++)
	{
		m_spPtr[i].distance = (histogram[(uchar)m_spPtr[i].avgColor]-minV)/(maxV- minV);

		//int k = m_centers[i].xy.x;
		//int j = m_centers[i].xy.y;
		//int radius = m_step;
		//if (1)
		//{
		//	for (int x = k- radius; x<= k+radius; x++)
		//	{
		//		for(int y = j - radius; y<= j+radius; y++)
		//		{
		//			if  (x<0 || x>m_width-1 || y<0 || y> m_height-1)
		//				continue;
		//			int idx = x+y*m_width;
		//			if (m_labels[idx] == i )
		//			{		
		//				grayMask.data[idx] = (uchar)m_spPtr[i].avgColor;
		//			}
		//		}
		//	}
		//}
	}
	cv::imwrite("maxHist.jpg",grayMask);
}
void MRFOptimize::GetSuperpixels(const unsigned char* mask, const uchar* featureMask)
{
	
	int size = m_width*m_height;
	for(int i=0; i<m_nPixel; i++)
	{		
		int k = m_centers[i].xy.x;
		int j = m_centers[i].xy.y;
		if (m_centers[i].nPoints ==0)			
		{
			m_spPtr[i].ps = 0;
			m_spPtr[i].avgColor = 0;
			m_spPtr[i].idx = i;
			m_spPtr[i].lable = i;
		}
		else
		{
			m_spPtr[i].avgColor = (m_centers[i].rgb.x+m_centers[i].rgb.y + m_centers[i].rgb.z)/3;
			m_spPtr[i].idx = i;
			m_spPtr[i].lable = i;

			float n = 0;			
			float nBGEdges(0);
			float nBgInliers(0);
			//以原来的中心点为中心，step +2　为半径进行更新
			int radius = m_step;
			for (int x = k- radius; x<= k+radius; x++)
			{
				for(int y = j - radius; y<= j+radius; y++)
				{
					if  (x<0 || x>m_width-1 || y<0 || y> m_height-1)
						continue;
					int idx = x+y*m_width;
					//std::cout<<idx<<std::endl;
					if (m_labels[idx] == i )
					{		
						
						if ( mask[idx] == 0xff && featureMask[idx] != 0xff)
							n++;
						if (featureMask[idx] == 0xff)
						{
							nBGEdges++;
						}
						else if(featureMask[idx] == 100)
						{
							nBgInliers++;
						}
					}
					//else if(isNeighbour(i,x,y,m_width,m_height,m_labels))
					//{
					//	if (find(m_spPtr[i].neighbors.begin(),m_spPtr[i].neighbors.end(),&m_spPtr[m_labels[idx]]) == m_spPtr[i].neighbors.end())
					//		m_spPtr[i].neighbors.push_back(&m_spPtr[m_labels[idx]]);
					//}
				}
			}
			//m_spPtr[i].ps  = min((max(n-nBGEdges-nBgInliers,0))/m_centers[i].nPoints,1.0f);
			m_spPtr[i].ps  = n/m_centers[i].nPoints;
		/*	if (nBGEdges > 0 )
				m_spPtr[i].ps = 0;
			else
				m_spPtr[i].ps  = min(n/m_centers[i].nPoints,1.f);*/
			//float c = 1e-10;
			//m_spPtr[i].ps *= (1-(nBgInliers)/(nfeatrues+nBgInliers+c));
		}
	}
	/*ProbImage(m_spPtr,m_labels,m_nPixel,m_width,m_height);
	NeighborsImage(m_spPtr,m_labels,m_nPixel,m_width,m_height,m_neighbor);*/
}
void MRFOptimize::GetSuperpixels(const unsigned char* mask, const uchar* featureMask,const float* distanceMask)
{
	
	int size = m_width*m_height;
	float avgDis =0 ;
	for(int i=0; i<m_nPixel; i++)
	{		
		int k = m_centers[i].xy.x;
		int j = m_centers[i].xy.y;
		if (m_centers[i].nPoints ==0)			
		{
			m_spPtr[i].ps = 0;
			m_spPtr[i].avgColor = 0;
			m_spPtr[i].idx = i;
			m_spPtr[i].lable = i;
			m_spPtr[i].distance = 0;
		}
		else
		{
			m_spPtr[i].avgColor = (m_centers[i].rgb.x+m_centers[i].rgb.y + m_centers[i].rgb.z)/3;
			m_spPtr[i].idx = i;
			m_spPtr[i].lable = i;

			float n = 0;			
			float nBGEdges(0);
			float nBgInliers(0);
			float distance = 0;
			int nZero(0);
			//以原来的中心点为中心，step +2　为半径进行更新
			int radius = m_step;
			for (int x = k- radius; x<= k+radius; x++)
			{
				for(int y = j - radius; y<= j+radius; y++)
				{
					if  (x<0 || x>m_width-1 || y<0 || y> m_height-1)
						continue;
					int idx = x+y*m_width;
					//std::cout<<idx<<std::endl;
					if (m_labels[idx] == i )
					{		
						if (distanceMask[idx] >0)
						{
							distance +=distanceMask[idx];
							nZero++;
						}
						if ( mask[idx] == 0xff)
							n++;
						if (featureMask[idx] == 0xff)
						{
							nBGEdges++;
						}
						else if(featureMask[idx] == 100)
						{
							nBgInliers++;
						}
					}
					//else if(isNeighbour(i,x,y,m_width,m_height,m_labels))
					//{
					//	if (find(m_spPtr[i].neighbors.begin(),m_spPtr[i].neighbors.end(),&m_spPtr[m_labels[idx]]) == m_spPtr[i].neighbors.end())
					//		m_spPtr[i].neighbors.push_back(&m_spPtr[m_labels[idx]]);
					//}
				}
			}
			//m_spPtr[i].ps  = min((max(n-nBGEdges-nBgInliers,0))/m_centers[i].nPoints,1.0f);
			if (abs(nZero) < 1e-6)
				m_spPtr[i].distance = 1;
			else
				m_spPtr[i].distance = distance/nZero;
			m_spPtr[i].ps  = n/m_centers[i].nPoints;
			avgDis += m_spPtr[i].distance;
			//m_spPtr[i].ps *=  distance;
			//if (m_spPtr[i].distance < 0.3 )
			//{
			//	m_spPtr[i].ps = 0;
			//	//std::cout<<" distance "<<m_spPtr[i].distance<<std::endl;
			//}
		/*	if (nBGEdges > 0 )
				m_spPtr[i].ps = 0;
			else
				m_spPtr[i].ps  = min(n/m_centers[i].nPoints,1.f);*/
			//float c = 1e-10;
			//m_spPtr[i].ps *= (1-(nBgInliers)/(nfeatrues+nBgInliers+c));
		}
	}
	avgDis/=m_nPixel;
	std::cout<<"avgDis "<<avgDis<<"\n";
	/*ProbImage(m_spPtr,m_labels,m_nPixel,m_width,m_height);
	NeighborsImage(m_spPtr,m_labels,m_nPixel,m_width,m_height,m_neighbor);*/
}
void MRFOptimize::GridCutOptimize(SuperPixel* spPtr, int num_pixels,float beta, int num_labels,const int width, const int height,int *result)
{
	
	typedef GRIDCUT::GridGraph_2D_4C<float,float,float> Grid;
	Grid* grid = new Grid(m_gWidth,m_gHeight);
	for (int y=0;y<m_gHeight;y++)
	{
		for (int x=0;x<m_gWidth;x++)
		{
			int i = y*m_gWidth + x;
			float d = min(1.0f,spPtr[i].ps*2);
			d = max(1e-20f,d);
			float d1 = -log(d);
			float d2 =  - log(1-d);
			grid->set_terminal_cap(grid->node_id(x,y),d1,d2);
			
			if (x<m_gWidth-1)
			{
				//float energy = (m_lmd1+m_lmd2*exp(-beta*abs(spPtr[i].avgColor-spPtr[y*w + x+1].avgColor)));
				float A = spPtr[i].avgColor-spPtr[y*m_gWidth + x+1].avgColor;
				float cap = (m_lmd1+m_lmd2*exp(-beta*abs(A)));			
				grid->set_neighbor_cap(grid->node_id(x,y),  +1,0,cap);
				grid->set_neighbor_cap(grid->node_id(x+1,y),-1,0,cap);
			}

			if (y<m_gHeight-1)
			{
				
				//float energy = (m_lmd1+m_lmd2*exp(-beta*abs(spPtr[i].avgColor-spPtr[(y+1)*w + x].avgColor)));
				float A = spPtr[i].avgColor-spPtr[y*m_gWidth + x+m_gWidth].avgColor;
				float cap = (m_lmd1+m_lmd2*exp(-beta*abs(A)));
				grid->set_neighbor_cap(grid->node_id(x,y),  0,+1,cap);
				grid->set_neighbor_cap(grid->node_id(x,y+1),0,-1,cap);
			}
		}
	}

	grid->compute_maxflow();

	for (int y=0;y<m_gHeight;y++)
	{
		for (int x=0;x<m_gWidth;x++)
		{
			if (grid->get_segment(grid->node_id(x,y))) 
				result[x + y*m_gWidth] = 1;
			else
				result[x + y*m_gWidth] = 0;
		}
	}
	delete grid;
}