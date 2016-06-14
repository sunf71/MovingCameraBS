#include "BlockWarping.h"
#include "findHomography.h"
#include <algorithm>
#include "CudaBSOperator.h"
class MyCompare
{
public:
	bool operator()(const Cell& a, const Cell& b)
	{
		return a.points0.size() > b.points0.size();
	}

};
float BlockWarping::MappingError(const cv::Mat& homo, const Points& p1, const Points& p2)
{
	Points tmp = p1;
	MatrixTimesPoints(homo,tmp);
	float err = 0;
	for(int i=0; i<p2.size(); i++)
	{
		float dx = tmp[i].x  -p2[i].x;
		float dy = tmp[i].y - p2[i].y;
		err += sqrt(dx*dx + dy*dy);
	}
	return err/p1.size();
}
float BlockWarping::MappingError(const double* ptr, const Points& p1, const Points& p2)
{
	Points tmp = p1;
	MatrixTimesPoints(ptr,tmp);
	float err = 0;
	for(int i=0; i<p2.size(); i++)
	{
		float dx = tmp[i].x  -p2[i].x;
		float dy = tmp[i].y - p2[i].y;
		err += sqrt(dx*dx + dy*dy);
	}
	return err;
}
float BlockWarping::MappingError(const std::vector<double>& homoVec, const Points& p1, const Points& p2)
{
	Points tmp = p1;
	MatrixTimesPoints(homoVec,tmp);
	float err = 0;
	for(int i=0; i<p2.size(); i++)
	{
		float dx = tmp[i].x  -p2[i].x;
		float dy = tmp[i].y - p2[i].y;
		err += sqrt(dx*dx + dy*dy);
	}
	return err/p1.size();
}
void BlockWarping::CalcBlkHomography()
{
	/*_blkInvHomos.resize(_blkSize);
	_blkHomos.resize(_blkSize);*/
	
	while (_emptyBlkIdx0.size() > 0)
	{
		_emptyBlkIdx1.clear();
		//std::random_shuffle(_emptyBlkIdx0.begin(),_emptyBlkIdx0.end());
		
		for(int e=0; e<_emptyBlkIdx0.size(); e++)
		{
			int i = _emptyBlkIdx0[e];
			/*if (i == 42)
				std::cout<<"	block "<< i<<"  has "<<_cells[i].points0.size()<<" features\n";*/
			if (_cells[i].points0.size() >= _minNumForHomo)
			{
				/*if (i == 42)
				std::cout<<"	block "<< i<<"  has "<<_cells[i].points0.size()<<" features, homography estimated\n";*/
				cv::Mat homo,invHomo;
				std::vector<uchar> inliers;
				//findHomographyDLT(f1[i],f0[i],homo);
				//findHomographyNormalizedDLT(f1[i],f0[i],homo);
				findHomographyEqa(_cells[i].points1,_cells[i].points0,homo);	
				
				invHomo = homo.inv();
				_blkErrors[_cells[i].idx] =MappingError(homo,_cells[i].points1,_cells[i].points0);
				//std::cout<<"error of cell "<<i<<" "<<_blkErrors[i]<<std::endl;

				//std::cout<<"direct setting cells "<<_cells[i].idx<<std::endl;
				//homo = cv::findHomography(f1[i],f0[i],inliers,CV_RANSAC,0.1);
				//std::cout<<homo<<"\n";
				
				memcpy(&_blkHomoVec[_cells[i].idx*8],homo.data,64);
				memcpy(&_blkInvHomoVec[_cells[i].idx*8],invHomo.data,64);
				/*_blkHomos[_cells[i].idx] = homo.clone();		
				_blkInvHomos[_cells[i].idx] = invHomo.clone();*/
			}	
			else if ( _cells[i].points1.size() > 0)
			{
				//有特征点但不足计算矩阵，在周围8邻域中若有某些邻域cell已经有homo，则选取一个误差最小的homo作为本cell的homo
				//若周围8邻域中都没有，则把周围8邻域内的特征点集合起来计算，作为这些cell的homo
				float minErr = 1e10;
				int minIdx = -1;
				int x = _cells[i].idx % _quadStep;
				int y = _cells[i].idx / _quadStep;
				Points f1,f0;
				std::vector<int> neighborIdx;
				for (int k = y -1;  k<= y+1; k++)
				{
					if ( k<0 || k >= _quadStep)
						continue;
					for(int j = x -1; j<= x+1; j++)
					{
						if ( j < 0 || j>= _quadStep)
							continue;
						int idx = k*_quadStep + j;
						neighborIdx.push_back(idx);
						for (int i=0; i<_cells[idx].points1.size(); i++)
						{
							f1.push_back(_cells[idx].points1[i]);
							f0.push_back(_cells[idx].points0[i]);
						}
						if (_blkErrors[idx]>0)
						{
							float err = MappingError(&_blkHomoVec[idx*8],_cells[i].points1,_cells[i].points0);
							if (err < minErr)
							{
								minErr = err;
								minIdx = idx;
							}					
						}
					}
				}
				if (minIdx >= 0)
				{
					/*if (i == 42)
					std::cout<<"min neighbor externel error setting cells "<<_cells[i].idx<<std::endl;*/
					/*_blkHomoVec[_cells[i].idx] = _blkHomoVec[minIdx];
					_blkInvHomoVec[_cells[i].idx] = _blkInvHomoVec[minIdx];*/
					int srcIdx = minIdx*8;
					int dstIdx = _cells[i].idx*8;
					memcpy(&_blkHomoVec[dstIdx],&_blkHomoVec[srcIdx],64);
					memcpy(&_blkInvHomoVec[dstIdx],&_blkInvHomoVec[srcIdx],64);
					_blkErrors[_cells[i].idx] = minErr;
				}
				else if(f1.size() >= _minNumForHomo)
				{
					cv::Mat homo,invHomo;
					findHomographyEqa(f1,f0,homo);	

					invHomo = homo.inv();
					_blkErrors[_cells[i].idx] =MappingError(homo,f1,f0);
					int dstIdx = _cells[i].idx*8;
					memcpy(&_blkHomoVec[dstIdx],homo.data,64);
					memcpy(&_blkInvHomoVec[dstIdx],invHomo.data,64);
					/*if (i == 42)
					std::cout<<"neighbor directs setting "<<_cells[i].idx<<std::endl;*/
					for (int t =0; t<neighborIdx.size(); t++)
					{
						int  dstIdx = neighborIdx[t]*8;
						
						memcpy(&_blkHomoVec[dstIdx],homo.data,64);
						memcpy(&_blkInvHomoVec[dstIdx],invHomo.data,64);
						_blkErrors[neighborIdx[t]] = _blkErrors[_cells[i].idx];
						//std::cout<<"neighbor directs setting "<<neighborIdx[t]<<std::endl;
					}
				}
				else
					_emptyBlkIdx1.push_back(i);
			}		
			else
			{
				//一个特征点没有，在周围8邻域中选择一个本cell内误差最小的
				float minErr = 1e10;
				int minIdx = -1;
				/*if (i==42)
					std::cout<<"cell 42 idx = "<<_cells[i].idx<<std::endl;*/
				int x = _cells[i].idx % _quadStep;
				int y = _cells[i].idx / _quadStep;
				for (int k = y -1;  k<= y+1; k++)
				{
					if ( k<0 || k >= _quadStep)
						continue;
					for(int j = x -1; j<= x+1; j++)
					{
						if ( j < 0 || j>= _quadStep)
							continue;
						int idx = k*_quadStep + j;
						/*if (i==42)
							std::cout<<"neighbor "<<idx<<" error "<<_blkErrors[idx]<<std::endl;*/
						if (_blkErrors[idx] > 0)
						{
							float err = _blkErrors[idx];
							if (err < minErr)
							{
								minErr = err;
								minIdx = idx;
							}					
						}
					}
				}
				
				if (minIdx >= 0)
				{
					int dstIdx = _cells[i].idx*8;
					int srcIdx = minIdx*8;
					memcpy(&_blkHomoVec[dstIdx],&_blkHomoVec[srcIdx],64);
					memcpy(&_blkInvHomoVec[dstIdx],&_blkInvHomoVec[srcIdx],64);
					
					_blkErrors[_cells[i].idx] = _blkErrors[minIdx];
					/*if (i == 42)
					std::cout<<"min  interner error setting cells "<<_cells[i].idx<<std::endl;*/
				}
				else
					_emptyBlkIdx1.push_back(i);
			}		
		}
		std::swap(_emptyBlkIdx1,_emptyBlkIdx0);
		/*std::cout<<_emptyBlkIdx0.size()<<std::endl;
		if (_emptyBlkIdx0.size() == 1)
			std::cout<<"--------only you "<<_emptyBlkIdx0[0]<<std::endl;*/
	}


}	

void BlockWarping::SetFeaturePoints(const Points& features1, const Points& features0)
{
	//将特征点归入到分块中
	for(int i=0; i< features1.size(); i++)
	{
		int idx = blockId(features1[i]);
		_cells[idx].points0.push_back(features1[i]);
		_cells[idx].points1.push_back(features0[i]);
		_cells[idx].idx = idx;
		/*_blkPoints0[idx].push_back(features1[i]);
		_blkPoints1[idx].push_back(features0[i]);*/
	}
	//std::sort(_cells.begin(),_cells.end(),MyCompare());
	//std::random_shuffle(_cells.begin(),_cells.end());

}

void BlockWarping::GpuWarp(const cv::Mat& img, cv::Mat& warpedImg)
{
	_dImg.upload(img);
	cudaMemcpy(_dBlkInvHomoVec,&_blkInvHomoVec[0],sizeof(double)*8*_blkSize,cudaMemcpyHostToDevice);
	cudaMemcpy(_dBlkHomoVec,&_blkHomoVec[0],sizeof(double)*8*_blkSize,cudaMemcpyHostToDevice);

	CudaWarp(_dImg, _quadStep,_dBlkHomoVec,_dBlkInvHomoVec,_dMapXY[0],_dMapXY[1], _dIMapXY[0],_dIMapXY[1],warpedImg);

	cv::gpu::merge(_dMapXY,2,_dMap);
	cv::gpu::merge(_dIMapXY,2,_dIMap);
	
}

void BlockWarping::GpuWarp(const cv::gpu::GpuMat& img, cv::gpu::GpuMat& warpedImg)
{
	
	cudaMemcpy(_dBlkInvHomoVec,&_blkInvHomoVec[0],sizeof(double)*8*_blkSize,cudaMemcpyHostToDevice);
	cudaMemcpy(_dBlkHomoVec,&_blkHomoVec[0],sizeof(double)*8*_blkSize,cudaMemcpyHostToDevice);

	CudaWarp(img, _quadStep,_dBlkHomoVec,_dBlkInvHomoVec,_dMapXY[0],_dMapXY[1], _dIMapXY[0],_dIMapXY[1],warpedImg);

	cv::gpu::merge(_dMapXY,2,_dMap);
	cv::gpu::merge(_dIMapXY,2,_dIMap);
	_dMap.download(_mapXY);
}

void BlockWarping::Warp(const cv::Mat& img, cv::Mat& warpedImg)
{
	for(int i=0; i<_mesh.rows-1; i++)
	{
		int * ptr0 = _mesh.ptr<int>(i);
		int * ptr1 = _mesh.ptr<int>(i+1);
		for(int j=0; j<_mesh.cols-1; j++)
		{
			int idx = i*_quadStep + j;
			int x0 = ptr0[2*j];
			int y0 = ptr0[2*j+1];
			cv::Point2i pt0(x0,y0);
			int x1 = ptr1[2*(j+1)];
			int y1 = ptr1[2*(j+1)+1];
			cv::Point2i pt1(x1,y1);
			cv::Rect roi(pt0,pt1);
			cv::Mat srcQuad = _srcPtMat(roi);
			cv::Mat dstQuad = _mapXY(roi);
			cv::Mat idstQuad = _invMapXY(roi);
			/*MatrixTimesMatPoints(_blkHomos[idx],srcQuad,dstQuad);
			MatrixTimesMatPoints(_blkInvHomos[idx],srcQuad,idstQuad);*/
			const double * hptr = &_blkHomoVec[idx*8];
			const double * ihptr = &_blkInvHomoVec[idx*8];
			for(int i=0; i<srcQuad.rows; i++)
			{
				const float* ptr = srcQuad.ptr<float>(i);
				float* dstPtr = dstQuad.ptr<float>(i);
				float* idstPtr = idstQuad.ptr<float>(i);
				for(int j=0; j<srcQuad.cols; j++)
				{
					float x = hptr[0]*ptr[2*j] + hptr[1]*ptr[2*j+1] + hptr[2];
					float y = hptr[3]*ptr[2*j] + hptr[4]*ptr[2*j+1] + hptr[5];
					float z = hptr[6]*ptr[2*j] + hptr[7]*ptr[2*j+1] + 1;
					dstPtr[2*j] = x/z;
					dstPtr[2*j+1] = y/z;			
					x = ihptr[0]*ptr[2*j] + ihptr[1]*ptr[2*j+1] + ihptr[2];
					y = ihptr[3]*ptr[2*j] + ihptr[4]*ptr[2*j+1] + ihptr[5];
					z = ihptr[6]*ptr[2*j] + ihptr[7]*ptr[2*j+1] +  1;
					idstPtr[2*j] = x/z;
					idstPtr[2*j+1] = y/z;
				}
			}
			
		}
	}
	//warpedImg = img.clone();
	cv::Mat maps[2];
	cv::split(_mapXY,maps);
	cv::remap(img,warpedImg,maps[0],maps[1],CV_INTER_CUBIC);
	
}

void BlockWarping::getFlow(cv::Mat& flow)
{
	flow.create(_mapXY.size(),CV_32FC2);
	for(int i=0; i<flow.rows; i++)
	{
		cv::Vec2f* ptr = _mapXY.ptr<cv::Vec2f>(i);
		
		cv::Vec2f* ptrFlow = flow.ptr<cv::Vec2f>(i);
		for(int j=0; j<flow.cols; j++)
		{
			ptrFlow[j] = cv::Vec2f(j-ptr[j][0],i-ptr[j][1]);
		}
	}
}

void GlobalWarping::getFlow(cv::Mat& flow)
{
	flow.create(_mapXY.size(),CV_32FC2);
	for(int i=0; i<flow.rows; i++)
	{
		cv::Vec2f* ptr = _mapXY.ptr<cv::Vec2f>(i);
		
		cv::Vec2f* ptrFlow = flow.ptr<cv::Vec2f>(i);
		for(int j=0; j<flow.cols; j++)
		{
			ptrFlow[j] = cv::Vec2f(j-ptr[j][0],i-ptr[j][1]);
		}
	}
}
void GlobalWarping::Warp(const cv::Mat& img, cv::Mat& warpedImg)
{
	_mapXY.create(img.rows,img.cols,CV_32FC2);
	_invMapXY.create(img.rows,img.cols,CV_32FC2);
	for(int i=0; i<img.rows; i++)
	{
		cv::Vec2f* dstPtr = _mapXY.ptr<cv::Vec2f>(i);
		cv::Vec2f* idstPtr = _invMapXY.ptr<cv::Vec2f>(i);
		for(int j=0; j<img.cols; j++)
		{
			cv::Point2f pt(j,i);
			MatrixTimesPoint(_homography,pt);
			dstPtr[j][0] = pt.x;
			dstPtr[j][1] = pt.y;
			pt.x = j, pt.y = i;
			MatrixTimesPoint(_invHomograpy,pt);
			idstPtr[j][0] = pt.x;
			idstPtr[j][1] = pt.y;
		}
	}
	cv::Mat mapXY[2];
	cv::split(_mapXY,&mapXY[0]);
	cv::remap(img,warpedImg,mapXY[0],mapXY[1],CV_INTER_CUBIC);
}
