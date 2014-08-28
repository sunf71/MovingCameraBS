#include "PictureHandler.h"
#include "SLIC.h"
#include "GCoptimization.h"
#include <math.h>
#include <hash_map>
#include <iostream>
#include <algorithm>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/video/tracking.hpp>

const float theta(0.35);
const float lmd1(0.3);
const float lmd2(3.0);

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

void ComputeAvgColor(SuperPixel* superpixels, size_t spSize, const int width, const int height, const unsigned char* imgData, const unsigned char* maskData)
{
	cv::Mat psMat(height,width,CV_8U);
	psMat = cv::Scalar(0);
	uchar* psImg = psMat.data;
	cv::Mat avgMat(height,width,CV_8U);
	avgMat = cv::Scalar(0);
	uchar* avgImg = avgMat.data;
	for(int i=0; i<spSize; i++)
	{
		float tmp = 0;
		float mtmp = 0;
		for( int j=0; j<superpixels[i].pixels.size(); j++)
		{
			int idx = (superpixels[i].pixels[j].first + superpixels[i].pixels[j].second*width);
			int idxC = idx*4;
			for (int c=1; c<4; c++)
				tmp += imgData[idxC+c]; 
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
	cv::imwrite("prob.jpg",psMat);
	cv::imwrite("avg.jpg",avgMat);
}

void GeneralGraph_DArraySArraySpatVarying(SuperPixel* spPtr, int num_pixels,float beta, int num_labels,const int width, const int height)
{
	int *result = new int[num_pixels];   // stores result of optimization

	// first set up the array for data costs
	int* data = new int[num_pixels*num_labels];
	
	for(int i=0; i<num_pixels; i++)
	{
		for(int j=0; j<num_labels; j++)
		{
			float d = min(1.0f,theta*spPtr[i].ps*2);
			d = max(0.00001f,d);
			float d1 = -log(d)*j;
			float d2 =  - log(1-d)*(1-j);
			data[i*num_labels + j] =(int)(d1+d2);
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
		gc->setSmoothCost(smooth);

		// now set up a grid neighborhood system
		for(int i=0; i<num_pixels; i++)
		{
			for(int j=0; j<spPtr[i].neighbors.size(); j++)
			{
				float energy = (lmd1+lmd2*exp(-beta*abs(spPtr[i].avgColor-spPtr[i].neighbors[j]->avgColor)));
				gc->setNeighbors(i,spPtr[i].neighbors[j]->idx,(int)energy);
			}
		}


		printf("\nBefore optimization energy is %d",gc->compute_energy());
		printf("\nBefore optimization  data energy is %d",gc->giveDataEnergy());
		printf("\nBefore optimization smooth energy is %d",gc->giveSmoothEnergy());
		gc->expansion(2);// run expansion for 2 iterations. For swap use gc->swap(num_iterations);
		printf("\nAfter optimization energy is %d",gc->compute_energy());
		printf("\nAfter optimization  data energy is %d",gc->giveDataEnergy());
		printf("\nAfter optimization smooth energy is %d",gc->giveSmoothEnergy());

		for ( int  i = 0; i < num_pixels; i++ )
			result[i] = gc->whatLabel(i);

		cv::Mat img(height,width,CV_8U);
		img = cv::Scalar(0);
		unsigned char* imgPtr = img.data;
		for(int i=0; i<num_pixels; i++)
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
		cv::imwrite("result.jpg",img);

		delete gc;
	}
	catch (GCException e){
		e.Report();
	}

	delete [] result;
	delete [] smooth;
	delete [] data;


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
	int *data = new int[num_pixels*num_labels];
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
	int *smooth = new int[num_labels*num_labels];
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
int main()
{
	/*testGCO();*/
	const int nLabels(2);
	
	using namespace std;
	const char filename[] = "..\\PTZ\\input0\\in000002.jpg";
	string maskFileName = "..\\result\\subsensem\\input0\\bin000002.png";
	string savename = "output";
	string saveLocation = ".\\";
	unsigned int* img = NULL;
	int width(0);
	int height(0);

	PictureHandler	picHand;
	picHand.GetPictureBuffer( string(filename), img, width, height );
	cv::Mat Img = cv::imread(filename);
	uchar* camData = new uchar[Img.total()*4];
	cv::Mat continuousRGBA(Img.size(), CV_8UC4, camData);	
	cv::cvtColor(Img,continuousRGBA,CV_BGR2RGBA,4);
	unsigned int* data = (unsigned*)camData;
	cv::Mat maskImg = cv::imread(maskFileName);
	cv::cvtColor(maskImg,maskImg,CV_BGR2GRAY);
	maskImg.convertTo(maskImg,CV_8U);
	const unsigned char* maskImgData = maskImg.data;
	
	
	size_t sz = width*height;
	int* labels = new int[sz];
	int numlabels(0);
	SLIC slic;
	slic.PerformSLICO_ForGivenK(data, width, height, labels, numlabels, 200, 20);//for a given number K of superpixels
	//slic.PerformSLICO_ForGivenStepSize(img, width, height, labels, numlabels, m_stepsize, m_compactness);//for a given grid step size
	//slic.DrawContoursAroundSegments(img, labels, width, height, 0);//for black contours around superpixels
	slic.DrawContoursAroundSegmentsTwoColors(data, labels, width, height);//for black-and-white contours around superpixels
	slic.SaveSuperpixelLabels(labels,width,height,savename+".dat",saveLocation);
	
	picHand.SavePicture(data, width, height, savename, saveLocation, 1, "_SLICO");// 0 is for BMP and 1 for JPEG)
	SuperPixel* spPtr = NULL;
	size_t spSize(0);
	GetSegment2DArray(spPtr,spSize,labels,width,height);
	ComputeAvgColor(spPtr,spSize,width,height,(unsigned char*)data,maskImgData);
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
	GeneralGraph_DArraySArraySpatVarying(spPtr,spSize,avgE,2,width,height);
	std::cout<<std::endl;
	std::cout<<spSize<<std::endl;
	std::cout<<spPtr[0].lable<<std::endl;
	std::cout<<spPtr[0].pixels.size()<<std::endl;
	std::cout<<spPtr[0].neighbors.size()<<std::endl;
	bool suc = true;
	for(int k=0; k<spSize; k++)
	{
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
	if(img) delete [] img;
	
	return 0;
}