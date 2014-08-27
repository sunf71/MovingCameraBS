#include "PictureHandler.h"
#include "SLIC.h"
#include <hash_map>
#include <iostream>
#include <algorithm>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/video/tracking.hpp>
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
	int lable;
	std::vector<Point2i> pixels;
	std::vector<SuperPixel*> neighbors;
	unsigned avgColor;
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
int main()
{
	using namespace std;
	const char filename[] = "input.jpg";
	string savename = "output";
	string saveLocation = ".\\";
	unsigned int* img = NULL;
	int width(0);
	int height(0);

	PictureHandler	picHand;
	picHand.GetPictureBuffer( string(filename), img, width, height );
	cv::Mat Img = cv::imread(filename);
	Img.convertTo(Img,CV_RGB2RGBA);
	unsigned int* data = (unsigned*)Img.data;
	for(int i=0; i<Img.cols*Img.rows; i++)
	{
		unsigned int ptr = data[i];
		data[i] = ((ptr)>>8)|(ptr<<24);
	}
	
	size_t sz = width*height;
	int* labels = new int[sz];
	int numlabels(0);
	SLIC slic;
	slic.PerformSLICO_ForGivenK(img, width, height, labels, numlabels, 200, 20);//for a given number K of superpixels
	//slic.PerformSLICO_ForGivenStepSize(img, width, height, labels, numlabels, m_stepsize, m_compactness);//for a given grid step size
	//slic.DrawContoursAroundSegments(img, labels, width, height, 0);//for black contours around superpixels
	slic.DrawContoursAroundSegmentsTwoColors(img, labels, width, height);//for black-and-white contours around superpixels
	slic.SaveSuperpixelLabels(labels,width,height,savename+".dat",saveLocation);
	
	picHand.SavePicture(img, width, height, savename, saveLocation, 1, "_SLICO");// 0 is for BMP and 1 for JPEG)
	SuperPixel* spPtr = NULL;
	size_t spSize(0);
	GetSegment2DArray(spPtr,spSize,labels,width,height);
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