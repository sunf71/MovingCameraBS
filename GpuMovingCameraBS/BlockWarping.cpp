#include "BlockWarping.h"
#include "findHomography.h"
#include <algorithm>
#include "CudaBSOperator.h"
#include <opencv2\gpu\gpu.hpp>
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
				//findHomographyEqa(f1[i],f0[i],homo);
				//findHomographyEqa(f1[i],f0[i],homo);
				findHomographyEqa(_cells[i].points1,_cells[i].points0,homo);	
				//homo = cv::findHomography(_cells[i].points1, _cells[i].points0, CV_LMEDS);
				/*std::cout << lhomo << "\n";
				std::cout << homo << "\n";*/
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
					//homo = cv::findHomography(f1, f0, CV_LMEDS);
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
	_dMap.download(_map);
	_dIMap.download(_invMap);
	cv::split(_map, _mapXY);
	cv::split(_invMap, _invMapXY);
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
			cv::Mat dstQuad = _map(roi);
			cv::Mat idstQuad = _invMap(roi);
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
	
	cv::split(_map,_mapXY);
	cv::remap(img, warpedImg, _mapXY[0], _mapXY[1], CV_INTER_CUBIC);
	
}

void BlockWarping::getFlow(cv::Mat& flow)
{
	flow.create(_map.size(),CV_32FC2);
	for(int i=0; i<flow.rows; i++)
	{
		cv::Vec2f* ptr = _map.ptr<cv::Vec2f>(i);
		
		cv::Vec2f* ptrFlow = flow.ptr<cv::Vec2f>(i);
		for(int j=0; j<flow.cols; j++)
		{
			ptrFlow[j] = cv::Vec2f(j-ptr[j][0],i-ptr[j][1]);
		}
	}
}
void GlobalWarping::SetFeaturePoints(const Points& p1, const Points& p2)
{
	//_homo = cv::findHomography( p2, p1,_inliers, CV_RANSAC, _threshold);
	findHomographyDLT((Points)p2, (Points)p1, _homo);
	_invHomo = _homo.inv();
	double* homoPtr = (double*)_homo.data;
	double* invHomoPtr = (double*)_invHomo.data;
	for (int i = 0; i < _height; i++)
	{
		cv::Vec2f* ptr = _map.ptr<cv::Vec2f>(i);
		cv::Vec2f* invPtr = _invMap.ptr<cv::Vec2f>(i);
		for (int j = 0; j < _width; j++)
		{
			float wx = homoPtr[0] * j + homoPtr[1] * i + homoPtr[2];
			float wy = homoPtr[3] * j + homoPtr[4] * i + homoPtr[5];
			float ww = homoPtr[6] * j + homoPtr[7] * i + homoPtr[8];
			wx /= ww;
			wy /= ww;
			ptr[j][0] = wx;
			ptr[j][1] = wy;
			wx = invHomoPtr[0] * j + invHomoPtr[1] * i + invHomoPtr[2];
			wy = invHomoPtr[3] * j + invHomoPtr[4] * i + invHomoPtr[5];
			ww = invHomoPtr[6] * j + invHomoPtr[7] * i + invHomoPtr[8];
			wx /= ww;
			wy /= ww;
			invPtr[j][0] = wx;
			invPtr[j][1] = wy;
		}
	}
	cv::split(_map, _mapXY);
	cv::split(_invMap, _invMapXY);
	_dMap.upload(_map);
	_dIMap.upload(_invMap);
	for (size_t i = 0; i < 2; i++)
	{
		_dIMapXY[i].upload(_invMapXY[i]);
		_dMapXY[i].upload(_mapXY[i]);
	}
	
	
	
}
void GlobalWarping::Warp(const cv::Mat& img, cv::Mat& warpedImg)
{
	cv::warpPerspective(img, warpedImg, _invHomo, img.size());
}

void GlobalWarping::GpuWarp(const cv::gpu::GpuMat& dimg, cv::gpu::GpuMat& dwimg)
{
	cv::gpu::warpPerspective(dimg, dwimg, _invHomo, dimg.size());
}

void GlobalWarping::getFlow(cv::Mat& flow)
{
	flow.create(_map.size(), CV_32FC2);
	for (int i = 0; i<flow.rows; i++)
	{
		cv::Vec2f* ptr = _map.ptr<cv::Vec2f>(i);

		cv::Vec2f* ptrFlow = flow.ptr<cv::Vec2f>(i);
		for (int j = 0; j<flow.cols; j++)
		{
			ptrFlow[j] = cv::Vec2f(j - ptr[j][0], i - ptr[j][1]);
		}
	}
}






void NBlockWarping::CalcBlkHomography()
{
	std::vector<bool> blkFlags(_blkSize);
	memset(&blkFlags[0], 0, sizeof(blkFlags));

	for (size_t i = 0; i < _blkSize; i++)
	{
		if (blkFlags[i])
			continue;

		cv::Mat homo, invHomo;
		Points f1, f2;
		AddFeaturePoints(f1, f2, i);
		//if the block hasn't got a homo and the features are enough, just calculate
		if (f1.size() > _minNumForHomo)
		{
			
			findHomographyEqa(f1, f2, homo);
			invHomo = homo.inv();
			memcpy(&_blkHomoVec[_cells[i].idx * 8], homo.data, 64);
			memcpy(&_blkInvHomoVec[_cells[i].idx * 8], invHomo.data, 64);
			//std::cout << i << " direct calculated\n";
			blkFlags[i] = true;
		}
		else
		{
			int x = i%_quadStep;
			int y = i / _quadStep;
			//acculate the neighbors untill get the minimu feature number
			int r = 1;
			std::vector<int> blockIds;
			blockIds.push_back(i); 
			while (f1.size() < _minNumForHomo)
			{
				for (int k = y - r; k <= y + r; k++)
				{
					if (k < 0 || k >= _quadStep)
						continue;
					if (k == y - r || k == y + r)
					{
						for (int j = x - r; j <= x + r; j++)
						{
							if (j < 0 || j >= _quadStep)
								continue;
							int idx = k*_quadStep + j;
							AddFeaturePoints(f1, f2, idx);
							blockIds.push_back(idx);
						}
					}
					else
					{
						if (x - r >= 0 && x - r < _quadStep)
						{
							int idx = k*_quadStep + x - r;
							AddFeaturePoints(f1, f2, idx);
							blockIds.push_back(idx);

						}
						if (x + r >= 0 && x + r < _quadStep)
						{
							int idx = k*_quadStep + x + r;
							AddFeaturePoints(f1, f2, idx);
							blockIds.push_back(idx);
						}
					}
					
				}				
				r++;
			}
			findHomographyEqa(f1, f2, homo);
			invHomo = homo.inv();
			//std::cout << "combination:";
			for (size_t i = 0; i < blockIds.size(); i++)
			{
				int id = blockIds[i];
				memcpy(&_blkHomoVec[_cells[id].idx * 8], homo.data, 64);
				memcpy(&_blkInvHomoVec[_cells[id].idx * 8], invHomo.data, 64);
				blkFlags[id] = true;
				//std::cout << id << ",";
			}
			//std::cout << "\n";
		}
	}
}

