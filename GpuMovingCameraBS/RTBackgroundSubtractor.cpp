#include "RTBackgroundSubtractor.h"
#include "CudaSuperpixel.h"
#include "RegionMerging.h"
#include "timer.h"
#include "FeaturePointRefine.h"
void RTBackgroundSubtractor::Initialize(cv::InputArray image)
{
	
	_img = image.getMat();
	_width = _img.cols;
	_height = _img.rows;
	_imgSize = _width*_height;
	_fgMask.create(_height,_width,CV_8U);
	_fgMask = cv::Scalar(0);
	cv::cvtColor(_img,_gray,CV_BGR2GRAY);
	if (_preGray.empty())
		_preGray = _gray.clone();
	//for test
	/*cv::Mat tmp = cv::imread("..\\moseg\\cars2\\in000001.jpg");
	cv::cvtColor(tmp,_preGray,CV_BGR2GRAY);*/
	
	_SPComputer = new SuperpixelComputer(_img.cols,_img.rows,_spStep,_spAlpha);
	_spWidth = _SPComputer->GetSPWidth();
	_spHeight = _SPComputer->GetSPHeight();
	_spSize = _spWidth*_spHeight;
	_segment = new int[_width*_height];

	_colorHists.resize(_spSize);
	_HOGs.resize(_spSize);
	_spPoses.resize(_spSize);
	_visited = new char[_spSize];
	_spMasks.resize(_spSize);
	for(int i=0; i<_spSize; i++)
	{
		_spMasks[i] = cv::Mat::zeros(_height,_width,CV_8U);
		_spPoses[i].clear();
		_colorHists[i].resize(_totalColorBins);
		_HOGs[i].resize(_hogBins);
		memset(&_HOGs[i][0],0,sizeof(float)*_hogBins);
		memset(&_colorHists[i][0],0,sizeof(float)*_totalColorBins);
	}

	_dFeatureDetector = new cv::gpu::GoodFeaturesToTrackDetector_GPU(100,0.01,_spStep);
}

void RTBackgroundSubtractor::operator()(cv::InputArray image, cv::OutputArray fgmask, double learningRate)
{
	_img = image.getMat();
	SaliencyMap();
	/*cv::Mat regionMask;
	GetRegionMap(regionMask);
	cv::imshow("regionMask", regionMask);*/
	fgmask.create(image.size(),CV_8UC1);
	MovingSaliency(fgmask.getMat());
	cv::swap(_preGray,_gray);
	cv::gpu::swap(_dGray,_dPreGray);
}
void RTBackgroundSubtractor::BuildHistogram(const int* labels, const SLICClusterCenter* centers)
{
	#pragma omp parallel for
	for(int i=0; i<_spSize; i++)
	{
		
		memset(&_HOGs[i][0],0,sizeof(float)*_hogBins);
		memset(&_colorHists[i][0],0,sizeof(float)*_totalColorBins);
		_spPoses[i].clear();

		int x = int(centers[i].xy.x+0.5);
		int y = int(centers[i].xy.y+0.5);
		
		for( int m=-_spStep+y; m<=_spStep+y; m++)
		{
			if (m<0 || m>= _height)
				continue;
			//cv::Vec3b* ptr = _img.ptr<cv::Vec3b>(m);
			float* magPtr = _magImg.ptr<float>(m);
			float* angPtr = _angImg.ptr<float>(m);
			//cv::Vec3f* labPtr = _labImg.ptr<cv::Vec3f>(m);
			cv::Vec3b* rgbPtr = _img.ptr<cv::Vec3b>(m);
			for(int n=-_spStep+x; n<=_spStep+x; n++)
			{
				if (n<0 || n>=_width)
					continue;
				int id = m*_width+n;
				_spMasks[i].data[id] = 0xff;
				if (labels[id] == i)
				{
					int bin = std::min<float>(floor(angPtr[n]/_hogStep),_hogBins-1);
					_HOGs[i][bin]+=magPtr[n];
					bin = 0;
					int s = 1;
					for(int c=0; c<3; c++)
					{
						//bin += s*std::min<float>(ceil((labPtr[n][c]-_colorMins[c]) /_colorSteps[c]),_colorBins-1);
						bin += s*std::min<float>(ceil((rgbPtr[n][c]-_colorMins[c]) /_colorSteps[c]),_colorBins-1);
						s*=_colorBins;
					}
					_colorHists[i][bin] ++;
					_spPoses[i].push_back(make_uint2(n,m));					
				}
			}
		}
	}
}

void RTBackgroundSubtractor::RegionMergingFast(const int* labels, const SLICClusterCenter* centers)
{
	//����ƽ�����ڳ����ؾ���֮��ľ���
	static const int dx4[] = {-1,0,1,0};
	static const int dy4[] = {0,-1,0,1};
	//const int dx8[8] = {-1, -1,  0,  1, 1, 1, 0, -1};
	//const int dy8[8] = { 0, -1, -1, -1, 0, 1, 1,  1};
	float avgCDist(0);
	float avgGDist(0);

	int nc(0);
   #pragma omp parallel for
	for (int idx=0; idx<_spSize; idx++)
	{
		int i = idx/_spWidth;
		int j = idx%_spWidth;
		for(int n=0; n<4; n++)
		{
			int dy = i+dy4[n];
			int dx = j+dx4[n];
			if (dy>=0 && dy<_spHeight && dx >=0 && dx < _spWidth)
			{
				int nIdx = dy*_spWidth + dx;
				nc++;
				avgCDist += cv::compareHist(_colorHists[idx],_colorHists[nIdx],CV_COMP_BHATTACHARYYA);
				avgGDist += cv::compareHist(_HOGs[idx],_HOGs[nIdx],CV_COMP_BHATTACHARYYA);

			}
		}
	}
	avgCDist/=nc;
	avgGDist/=nc;
	float confidence = (avgGDist)/(avgCDist+avgGDist);
	//rgbHConfidence = (avgGDist)/(avgDist+avgGDist);
	float threshold = ((avgCDist*confidence+(1-confidence)*avgGDist));	
	//std::cout<<"threshold: "<<threshold<<std::endl;
	//std::ofstream file("mergeOut.txt");
	

	float pixDist(0);	
	_regSizes.clear();
	_regIdices.clear();
	int regSize(0);
	//��ǰ�±�ǩ
	int curLabel(0);
	memset(_visited ,0,_spSize);
	memset(_segment,0,sizeof(int)*_spSize);
	std::vector<int> singleLabels;
	//region growing �����label
	std::vector<int> newLabels(_spSize);
	
	//nih::Timer timer;
	//timer.start();
	std::set<int> boundarySet;
	boundarySet.insert(rand()%_spSize);
	//boundarySet.insert(3);
	//boundarySet.insert(190);
	std::vector<int> labelGroup;
	
	while(!boundarySet.empty())
	{
		//std::cout<<boundarySet.size()<<std::endl;
		labelGroup.clear();
		std::set<int>::iterator itr = boundarySet.begin();
		int label = *itr;
		//file<<"seed: "<<label<<"\n";
		_visited[label] = true;
		labelGroup.push_back(label);		
		//newLabels[label] = curLabel;
		boundarySet.erase(itr);
		SLICClusterCenter cc = centers[label];
		int k = cc.xy.x;
		int j = cc.xy.y;		
		float4 regColor = cc.rgb;
		int ix = label%_spWidth;
		int iy = label/_spWidth;
		pixDist = 0;
		regSize = 1;
		
		RegInfos neighbors, tneighbors;
		while(pixDist < threshold && regSize<_spSize)
		{
			//file<<"iy:"<<iy<<"ix:"<<ix<<"\n";
			
			for(int d=0; d<4; d++)
			{
				int x = ix+dx4[d];
				int y = iy + dy4[d];
				size_t idx = x+y*_spWidth;
				if (x>=0 && x<_spWidth && y>=0 && y<_spHeight && !_visited[idx])
				{
					_visited[idx] = true;
					float rd = cv::compareHist(_colorHists[idx],_colorHists[label],CV_COMP_BHATTACHARYYA);
					float hd = cv::compareHist(_HOGs[idx],_HOGs[label],CV_COMP_BHATTACHARYYA);
					float dist = confidence*rd + 	hd*(1-confidence);
					neighbors.push(RegInfo(idx,x,y,dist));
				}
			}
			
			if (neighbors.empty())
				break;
			RegInfo sp = neighbors.top();
			pixDist = sp.dist;
			
			int minIdx = sp.label;
			ix = sp.x;
			iy = sp.y;
			if (pixDist < threshold)
			{
				neighbors.pop();
				//file<<"nearst neighbor "<<minIdx<<"("<<iy<<","<<ix<<") with distance:"<<pixDist<<"\n";
				float tmpx = regColor.x;
				float tmpy = regColor.y;
				float tmpz = regColor.z;
				regColor.x = (regColor.x*regSize + centers[minIdx].rgb.x)/(regSize+1);
				regColor.y = (regColor.y*regSize + centers[minIdx].rgb.y)/(regSize+1);
				regColor.z = (regColor.z*regSize + centers[minIdx].rgb.z)/(regSize+1);
				float t = 2.0;
				float dx = abs(tmpx - regColor.x);
				float dy = abs(tmpy - regColor.y);
				float dz = abs(tmpz - regColor.z);
			
				/*regColor.x += centers[minIdx].rgb.x;
				regColor.y += centers[minIdx].rgb.y;
				regColor.z += centers[minIdx].rgb.z;*/
				regSize++;
				labelGroup.push_back(minIdx);
				

				for(int i=0; i<_totalColorBins; i++)
				{
					_colorHists[label][i] += _colorHists[minIdx][i];
				}
				cv::normalize(_colorHists[label],_colorHists[label],1,0,cv::NORM_L1 );
		
				for(int i=0; i<_hogBins; i++)
				{
					_HOGs[label][i] += _HOGs[minIdx][i];
				}
				cv::normalize(_HOGs[label],_HOGs[label],1,0,cv::NORM_L1 );
				_visited[minIdx] = true;
				if (sqrt(dx*dx +dy*dy +dz*dz) > t)
				{
					while(!tneighbors.empty())
						tneighbors.pop();
					while(!neighbors.empty())
					{
						RegInfo sp = neighbors.top();
						neighbors.pop();
						float rd = cv::compareHist(_colorHists[sp.label],_colorHists[label],CV_COMP_BHATTACHARYYA);
						float hd = cv::compareHist(_HOGs[sp.label],_HOGs[label],CV_COMP_BHATTACHARYYA);
						sp.dist =  confidence*rd + 	hd*(1-confidence);
						tneighbors.push(sp);
					}
					std::swap(neighbors,tneighbors);
				}
				/*segmented[minIdx] = k;*/
				//result.data[minIdx] = 0xff;
				//smask.data[minIdx] = 0xff;
				
				std::set<int>::iterator itr =boundarySet.find(minIdx);
				if ( itr!= boundarySet.end())
				{
					boundarySet.erase(itr);
				}
			}
			/*else
			{			
				file<<"nearst neighbor "<<minIdx<<"("<<iy<<","<<ix<<") with distance:"<<pixDist<<"overpass threshold "<<regMaxDist<<"\n";
			}*/
		}
		_nColorHists.push_back(_colorHists[label]);	
		_nHOGs.push_back(_HOGs[label]);
		_regIdices.push_back(labelGroup);

		for(int i=0; i<labelGroup.size(); i++)
		{
			newLabels[labelGroup[i]] = curLabel;
		}

		_regSizes.push_back(regSize);
		curLabel++;		
		std::vector<RegInfo> *vtor = (std::vector<RegInfo> *)&neighbors;		
		for(int i=0; i<vtor->size(); i++)
		{
			int label = ((RegInfo)vtor->operator [](i)).label;
			_visited[label] = false;
			if (boundarySet.find(label) == boundarySet.end())
				boundarySet.insert(label);			
		}
		if (regSize <2)
			singleLabels.push_back(label);
	}
	
	
	//�Ե��������أ�������Ƿ��ڴ�����֮�У��ܱ���������labelһ����
	for (int i=0; i<singleLabels.size(); i++)
	{
		int label = singleLabels[i];
		int ix = label%_spWidth;
		int iy = label/_spWidth;
		std::vector<int> ulabel;
		
		for(int d=0; d<4; d++)
		{
			int x = ix+dx4[d];
			int y = iy + dy4[d];
			if (x>=0 && x<_spWidth && y>=0 && y<_spHeight)
			{
				int nlabel = x+y*_spWidth;		
				if (std::find(ulabel.begin(),ulabel.end(),newLabels[nlabel]) == ulabel.end())
					ulabel.push_back(newLabels[nlabel]);
			}
			
		}
		if (ulabel.size()<=2)
		{
				newLabels[label] = ulabel[0];
				_regSizes[ulabel[0]]++;
		}
	}
	
	for(int i=0; i<newLabels.size(); i++)
	{
		#pragma omp parallel for
		for (int j=0; j<_spPoses[i].size(); j++)
			_segment[_spPoses[i][j].x + _spPoses[i][j].y*_width] = newLabels[i];
	}

	
}
void RTBackgroundSubtractor::SaliencyMap()
{	
	
#ifndef REPORT
	nih::Timer timer;
	timer.start();
	std::cout<<"----------\n";
#endif
	_dCurrFrame.upload(_img);
	cv::gpu::cvtColor(_dCurrFrame,_dGray,CV_BGR2GRAY);
	if (_dPreGray.empty())
		_dPreGray = _dGray.clone();

	_img.convertTo(_fImg,CV_32FC3,1.0/255);
	//cv::cvtColor(_fImg,_labImg,CV_BGR2Lab);
#ifndef REPORT
	timer.stop();
	std::cout<<"	cvtColor "<<timer.seconds()*1000<<" ms\n";
#endif	
	
	

#ifndef REPORT
	timer.start();
#endif
	//superpixel
	_SPComputer->ComputeBigSuperpixel(_img);
#ifndef REPORT
	timer.stop();
	std::cout<<"	Superpixel "<<timer.seconds()*1000<<" ms\n";
#endif	
	int * labels;
	SLICClusterCenter* centers;
	int num(0);
	_SPComputer->GetSuperpixelResult(num,labels,centers);

	//
	//build histogram
#ifndef REPORT
	timer.start();
#endif	
	
	cv::cvtColor(_img,_gray,CV_BGR2GRAY);
	cv::GaussianBlur(_gray,_gray,cv::Size(3,3),0);
	
	cv::Mat dx,dy,ang,mag;
	cv::Scharr(_gray,_dxImg,CV_32F,1,0);
	cv::Scharr(_gray,_dyImg,CV_32F,0,1);
	cv::cartToPolar(_dxImg,_dyImg,_magImg,_angImg,true);
	BuildHistogram(labels,centers);
#ifndef REPORT
	timer.stop();
	std::cout<<"	Build histogram "<<timer.seconds()*1000<<" ms\n";
#endif

#ifndef REPORT
	timer.start();
#endif	
	//region merging
	RegionMergingFast(labels,centers);
#ifndef REPORT
	timer.stop();
	std::cout<<"	Region Merging "<<timer.seconds()*1000<<" ms\n";
#endif	


}



void RTBackgroundSubtractor::MovingSaliency(cv::Mat& fgMask)
{
#ifndef REPORT
	nih::Timer timer;
	timer.start();
#endif	
	int *labels(NULL), num(0);
	SLICClusterCenter* centers(NULL);
	_SPComputer->GetSuperpixelResult(num,labels,centers);
	fgMask = cv::Scalar(0);

	//���������С��threashold �������еĳ����ؽ���klt����	
	int regThreshold = 15;
	std::vector<cv::Point2f> pt0,pt1;
	////��ÿ�������ؽ���������ȡ��ÿ����������ȡ1��������
	//pt0.resize(_spSize);
	//#pragma omp parallel for
	//for(int i=0; i<_spSize; i++)
	//{
	//	cv::goodFeaturesToTrack(_gray,pt1,1,0.05,5,_spMasks[i]);
	//	//for(int j=0; j<1; j++)
	//		pt0[i] = pt1[0];
	//}
	//timer.stop();
	//std::cout<<"feature extracting "<<timer.seconds()*1000<<" ms \n";


	std::vector<int> regs;
	std::vector<int> sps;
	std::vector<std::vector<cv::Point2f>> spfeatures;
	for(int i=0; i<_regSizes.size(); i++)
	{
		if (_regSizes[i] < regThreshold)
		{
			
			for(int j=0; j<_regIdices[i].size(); j++)
			{				
				float2 xy = centers[_regIdices[i][j]].xy;
				pt0.push_back(cv::Point2f(xy.x,xy.y));
				sps.push_back(_regIdices[i][j]);
				regs.push_back(i);
			}
			
		}
	}
	//��С�����еĳ��������ĵ���ӵ�pt0
	for(int i=0; i<spfeatures.size(); i++)
	{
		for(int j=0; j<spfeatures[i].size(); j++)
			pt0.push_back(spfeatures[i][j]);
	}
	upload(pt0,_dSPCenters);
	std::vector<float> dists(pt0.size());
	
	

	
#ifndef REPORT
	//��ʾ�������
	/*cv::Mat mask;
	MatchingResult(_gray,_preGray,pt0,pt1,mask);
	cv::imshow("klt tracking", mask);
	cv::waitKey();*/
#endif


	//������˶����бȽ�
	int nf(0);
	cv::Mat homography;
	//good features
	std::vector<cv::Point2f> vf1,vf2;
	std::vector<cv::Point2f> tf1,tf2;
	std::vector<uchar> status;
	std::vector<float> err;
	std::vector<uchar> inliers;
	//cv::goodFeaturesToTrack(_gray, vf1, 100, 0.01, _spStep);	
	/*_dCurrPts.create(1,_dFeatures.cols+_dSPCenters.cols,CV_32FC2);
	_dFeatures.copyTo(*/


	_dFeatureDetector->operator()(_dGray,_dFeatures);

#ifndef REPORT
	timer.stop();
	std::cout<<"	gpu good features to track "<<timer.seconds()*1000<<" ms\n";
#endif	

#ifndef REPORT
	timer.start();
#endif	
	
	download(_dFeatures,vf1);
	nf = vf1.size();
	vf2.resize(nf);
	tf1.resize(vf1.size() + pt0.size());
	memcpy(&tf1[0],&vf1[0],sizeof(cv::Point2f)*nf);
	memcpy(&tf1[vf1.size()],&pt0[0],sizeof(cv::Point2f)*pt0.size());
	upload(tf1,_dCurrPts);
	d_pyrLk.sparse(_dGray,_dPreGray,_dCurrPts,_dPrevPts,d_Status);
	download(d_Status,status);
	download(_dPrevPts,tf2);
#ifndef REPORT
	timer.stop();
	std::cout<<"	gpu optical flow "<<timer.seconds()*1000<<" ms\n";
#endif	
#ifndef REPORT
	timer.start();
#endif	
	int k(0);
	for(int i=0; i<nf; i++)
	{
		if (status[i] == 1)
		{
			vf1[k] = vf1[i];
			vf2[k] = tf2[i];
			k++;
		}
		
	}
	vf1.resize(k);
	vf2.resize(k);
	homography = cv::findHomography(vf1,vf2,inliers,CV_RANSAC,1.0);

#ifndef REPORT
	timer.stop();
	std::cout<<"	RANSAC Motion Estimation "<<timer.seconds()*1000<<" ms\n";
#endif	

#ifndef REPORT
	timer.start();
#endif	
	double * homoPtr = (double*)homography.data;

	std::vector<int> fgRegs;
	float avgDist(0);
	for(int i=nf; i<tf1.size(); i++)
	{
		if (status[i] == 1)
		{
			float wx = homoPtr[0]*tf1[i].x + homoPtr[1]*tf1[i].y + homoPtr[2];
			float wy = homoPtr[3]*tf1[i].x + homoPtr[4]*tf1[i].y + homoPtr[5];
			float w = homoPtr[6]*tf1[i].x + homoPtr[7]*tf1[i].y + homoPtr[8];
			wx /= w;
			wy /= w;
			float dx = tf2[i].x - wx;
			float dy = tf2[i].y - wy;
			float dist = sqrt(dx*dx + dy*dy);	
			dists[i-nf] = dist;			
		}
	}
	
	
	for(int i=0; i<pt0.size(); i++)
	{
		if (dists[i] > _mThreshold)
			dists[i] = 255;
		else
			dists[i] = 0;
	}	
#ifndef REPORT
	timer.stop();
	std::cout<<"	Motion Verify "<<timer.seconds()*1000<<" ms\n";
#endif
	
	//cv::Mat distMask(_height,_width,CV_8U);
	//distMask = cv::Scalar(0);
	//for(int i=0; i<fgRegs.size(); i++)
	//{
	//	for(int r =0; r<_regIdices[fgRegs[i]].size(); r++)
	//	{
	//		for(int j=0; j<_spPoses[_regIdices[fgRegs[i]][r]].size(); j++)
	//		{
	//			int idx = _spPoses[_regIdices[fgRegs[i]][r]][j].x + _spPoses[_regIdices[fgRegs[i]][r]][j].y*_width;
	//			fgMask.data[idx] = 0xff;
	//			
	//		}
	//	}
	//}
#ifndef REPORT
	timer.start();
#endif
	cv::Mat spFGMask(_spHeight,_spWidth,CV_8U);
	spFGMask = cv::Scalar(0);
	for (int i=0; i<sps.size(); i++)
	{
		spFGMask.data[sps[i]] =  dists[i];
		/*for(int j=0; j<_spPoses[sps[i]].size(); j++)
		{
			int idx = _spPoses[sps[i]][j].x + _spPoses[sps[i]][j].y * _width;
			fgMask.data[idx] = dists[i];
		}*/
	}
	//std::cout<<"before blur \n"<<spFGMask<<"\nafter blur\n";
	cv::blur(spFGMask,spFGMask,cv::Size(3,3),cv::Point(-1,-1),cv::BORDER_CONSTANT);
	//std::cout<<spFGMask<<"\n";
	cv::threshold(spFGMask,spFGMask,57,255,cv::THRESH_BINARY);
	//std::cout<<"after threshold \n"<<spFGMask<<"\n";
	for (int i=0; i<sps.size(); i++)
	{
		if (spFGMask.data[sps[i]] ==255)
		{
			for(int j=0; j<_spPoses[sps[i]].size(); j++)
			{
				int idx = _spPoses[sps[i]][j].x + _spPoses[sps[i]][j].y * _width;
				fgMask.data[idx] = dists[i];
			}
		}
	}
#ifndef REPORT
	timer.stop();
	std::cout<<"	Output Mask "<<timer.seconds()*1000<<" ms\n";
#endif
	//cv::Mat regMask(_height,_width,CV_8U);
	//regMask = cv::Scalar(0);
	//for(int i=0; i<_spPoses[9*16].size(); i++)
	//{
	//	int idx = _spPoses[9*16][i].x + _spPoses[9*16][i].y*_width;
	//	regMask.data[idx] = 0xff;
	//}
	//cv::goodFeaturesToTrack(_gray,pt0,5,0.5,5,regMask);
	//cv::calcOpticalFlowPyrLK(_gray,_preGray,pt0,pt1,status,err);
	//for(int i=0; i<pt0.size(); i++)
	//{
	//
	//	
	//	if (status[i] == 1)
	//	{
	//		float wx = homoPtr[0]*pt0[i].x + homoPtr[1]*pt0[i].y + homoPtr[2];
	//		float wy = homoPtr[3]*pt0[i].x + homoPtr[4]*pt0[i].y + homoPtr[5];
	//		float w = homoPtr[6]*pt0[i].x + homoPtr[7]*pt0[i].y + homoPtr[8];
	//		wx /= w;
	//		wy /= w;
	//		float dx = pt1[i].x - wx;
	//		float dy = pt1[i].y - wy;
	//		float dist = sqrt(dx*dx + dy*dy);			
	//		std::cout<<dist<<std::endl;
	//	}
	//	
	//}
	//MatchingResult(_gray,_preGray,pt0,pt1,mask);
	//cv::imshow("klt tracking with mask", mask);

	//cv::imshow("region mask",regMask);
	//cv::imshow("saliency map",fgMask);
	//cv::imshow("dist map", distMask);
}
void RTBackgroundSubtractor::GetSuperpixelMap(cv::Mat& sp)
{
	_SPComputer->GetVisualResult(_img,sp);
}

void RTBackgroundSubtractor::GetRegionMap(cv::Mat& mask)
{
	mask.create(_height,_width,CV_8UC3);
	std::vector<int> color(_spSize);
	CvRNG rng= cvRNG(cvGetTickCount());
	for(int i=0; i<_spSize;i++)
		color[i] = cvRandInt(&rng);
	// Draw random color
	for(int i=0;i<_height;i++)
	{
		cv::Vec3b* ptr = _img.ptr<cv::Vec3b>(i);
	
		for(int j=0;j<_width;j++)
		{ 
			int cl = _segment[i*_width+j];
			((uchar *)(mask.data + i*mask.step.p[0]))[j*mask.step.p[1] + 0] = (color[cl])&255;
			((uchar *)(mask.data + i*mask.step.p[0]))[j*mask.step.p[1] + 1] = (color[cl]>>8)&255;
			((uchar *)(mask.data + i*mask.step.p[0]))[j*mask.step.p[1] + 2] = (color[cl]>>16)&255;
		
		}
	}
}