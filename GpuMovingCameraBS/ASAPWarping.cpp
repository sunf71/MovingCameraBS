#include "ASAPWarping.h"
#include "SparseSolver.h"
#include <fstream>
#include <opencv\cv.h>
#include "findHomography.h"
#include "CudaBSOperator.h"
bool isPointInTriangular(const cv::Point2f& pt, const cv::Point2f& V0, const cv::Point2f& V1, const cv::Point2f& V2)
{   
	float lambda1 = ((V1.y-V2.y)*(pt.x-V2.x) + (V2.x-V1.x)*(pt.y-V2.y)) / ((V1.y-V2.y)*(V0.x-V2.x) + (V2.x-V1.x)*(V0.y-V2.y));
	float lambda2 = ((V2.y-V0.y)*(pt.x-V2.x) + (V0.x-V2.x)*(pt.y-V2.y)) / ((V2.y-V0.y)*(V1.x-V2.x) + (V0.x-V2.x)*(V1.y-V2.y));
	float lambda3 = 1-lambda1-lambda2;
	if (lambda1 >= 0.0 && lambda1 <= 1.0 && lambda2 >= 0.0 && lambda2 <= 1.0 && lambda3 >= 0.0 && lambda3 <= 1.0)
		return true;
	else
		return false;

}
void ASAPWarping::CreateSmoothCons(std::vector<float> weights)
{
	_rowCount = 0;
	int i=0, j=0;
	addCoefficient_5(i,j,weights[0]);
	addCoefficient_6(i,j,weights[0]);
	int idx;
	i=0;j=_width-1;
	idx = _quadStep - 1;
	addCoefficient_7(i,j,weights[idx]);
	addCoefficient_8(i,j,weights[idx]);

	i=_height-1;j=0;
	idx = _quadStep*_quadStep - _quadStep;
	addCoefficient_3(i,j,weights[idx]);
	addCoefficient_4(i,j,weights[idx]);

	i=_height-1;j=_width-1;
	idx = _quadStep*_quadStep-1;
	addCoefficient_1(i,j,weights[idx]);
	addCoefficient_2(i,j,weights[idx]);

	i=0;
	for(j=1; j<=_width-2; j++)
	{
		addCoefficient_5(i,j,weights[j]);
		addCoefficient_6(i,j,weights[j]);
		addCoefficient_7(i,j,weights[j]);
		addCoefficient_8(i,j,weights[j]);
	}

	i=_height-1;
	for(j=1; j<=_width-2; j++)
	{
		idx = j+_quadStep*(_quadStep-1);
		addCoefficient_1(i,j,weights[idx]);
		addCoefficient_2(i,j,weights[idx]);
		addCoefficient_3(i,j,weights[idx]);
		addCoefficient_4(i,j,weights[idx]);
	}

	j=0;
	for( i=1; i<=_height-2; i++)
	{
		idx = i*_quadStep;
		addCoefficient_3(i,j,weights[idx]);
		addCoefficient_4(i,j,weights[idx]);
		addCoefficient_5(i,j,weights[idx]);
		addCoefficient_6(i,j,weights[idx]);
	}

	j=_width-1;
	for( i=1; i<=_height-2; i++)
	{
		idx = j+i*_quadStep-1;
		addCoefficient_1(i,j,weights[idx]);
		addCoefficient_2(i,j,weights[idx]);
		addCoefficient_7(i,j,weights[idx]);
		addCoefficient_8(i,j,weights[idx]);
	}

	for (i=1; i<= _height-2; i++)
	{
		for (j=1; j<= _width-2;j++)
		{
			idx = (i-1)*_quadStep + j-1;
			addCoefficient_1(i,j,weights[idx]);
			addCoefficient_2(i,j,weights[idx]);

			idx = (i-1)*_quadStep + j-1;
			addCoefficient_3(i,j,weights[idx]);
			addCoefficient_4(i,j,weights[idx]);

			idx = i*_quadStep+j;
			addCoefficient_5(i,j,weights[idx]);
			addCoefficient_6(i,j,weights[idx]);

			idx = i*_quadStep + j-1;
			addCoefficient_7(i,j,weights[idx]);
			addCoefficient_8(i,j,weights[idx]);
		}
	}
	_sRowCount = _rowCount;
}
void ASAPWarping::CreateSmoothCons(float weight)
{
	       _rowCount = 0;
           int i=0, j=0;
           addCoefficient_5(i,j,weight);
           addCoefficient_6(i,j,weight);
           
           i=0;j=_width-1;
           addCoefficient_7(i,j,weight);
           addCoefficient_8(i,j,weight);
          
           i=_height-1;j=0;
           addCoefficient_3(i,j,weight);
           addCoefficient_4(i,j,weight);
           
           i=_height-1;j=_width-1;
           addCoefficient_1(i,j,weight);
           addCoefficient_2(i,j,weight);
           
           i=0;
           for(j=1; j<=_width-2; j++)
		   {
              addCoefficient_5(i,j,weight);
              addCoefficient_6(i,j,weight);
              addCoefficient_7(i,j,weight);
              addCoefficient_8(i,j,weight);
		   }
           
           i=_height-1;
           for(j=1; j<=_width-2; j++)
           {
			  addCoefficient_1(i,j,weight);
              addCoefficient_2(i,j,weight);
              addCoefficient_3(i,j,weight);
              addCoefficient_4(i,j,weight);
		   }
           
           j=0;
           for( i=1; i<=_height-2; i++)
		   {
              addCoefficient_3(i,j,weight);
              addCoefficient_4(i,j,weight);
              addCoefficient_5(i,j,weight);
              addCoefficient_6(i,j,weight);
		   }
           
           j=_width-1;
            for( i=1; i<=_height-2; i++)
		   {
              addCoefficient_1(i,j,weight);
              addCoefficient_2(i,j,weight);
              addCoefficient_7(i,j,weight);
              addCoefficient_8(i,j,weight);
			}
           
           for (i=1; i<= _height-2; i++)
		   {
              for (j=1; j<= _width-2;j++)
			  {
              addCoefficient_1(i,j,weight);
              addCoefficient_2(i,j,weight);
              addCoefficient_3(i,j,weight);
              addCoefficient_4(i,j,weight);
              addCoefficient_5(i,j,weight);
              addCoefficient_6(i,j,weight);
              addCoefficient_7(i,j,weight);
              addCoefficient_8(i,j,weight);
			  }
		   }
		   _sRowCount = _rowCount;
}

void ASAPWarping::SetControlPts(std::vector<cv::Point2f>& inputsPts, std::vector<cv::Point2f>& outputsPts)
{
	int len = inputsPts.size();
	_dataterm_element_orgPt = inputsPts;
	_dataterm_element_desPt = outputsPts;

	_dataterm_element_i = std::vector<float>(len,0);
	_dataterm_element_j = std::vector<float>(len,0);

	_dataterm_element_V00 = std::vector<float>(len,0);
	_dataterm_element_V01 = std::vector<float>(len,0);
	_dataterm_element_V10 = std::vector<float>(len,0);
	_dataterm_element_V11 = std::vector<float>(len,0);

	std::vector<float> coefficients;
	for(int i=0; i<len; i++)
	{
		//std::cout<<i<<std::endl;
		cv::Point2f pt = inputsPts[i];
		_dataterm_element_i[i] = min(_height-1,(int)(pt.y + 0.5)/(_quadHeight) +1) ;
		_dataterm_element_j[i] = min(_width-1,(int)(pt.x + 0.5)/(_quadWidth) + 1);

		Quad qd = _source->getQuad(_dataterm_element_i[i],_dataterm_element_j[i]);


		
		qd.getBilinearCoordinates(pt,coefficients);
		_dataterm_element_V00[i] = coefficients[0];
		_dataterm_element_V01[i] = coefficients[1];
		_dataterm_element_V10[i] = coefficients[2];
		_dataterm_element_V11[i] = coefficients[3];
	}

}

void ASAPWarping::CreateDataCons(cv::Mat& b)
{
	int len = _dataterm_element_i.size();
	_num_data_cons = len*2;
	_DataConstraints.create(_num_data_cons*4,3,CV_32F);
	b.create(_num_data_cons+_num_smooth_cons,1,CV_32F);
	b = cv::Scalar(0);
	_DCc = 0;
	for(int k=0; k<len; k++)
	{
		//std::cout<<k<<std::endl;
		float i = _dataterm_element_i[k];
		float j = _dataterm_element_j[k];
		float v00 = _dataterm_element_V00[k];
		float v01 = _dataterm_element_V01[k];
		float v10 = _dataterm_element_V10[k];
		float v11 = _dataterm_element_V11[k];
		float * ptr = _DataConstraints.ptr<float>(_DCc);       
		ptr[0] = _rowCount; ptr[1] = _x_index[(i-1)*_width+j-1]; ptr[2]= v00; _DCc = _DCc+1;
		ptr = _DataConstraints.ptr<float>(_DCc); 
		ptr[0] = _rowCount; ptr[1] = _x_index[(i-1)*_width+j]; ptr[2]= v01; _DCc = _DCc+1;
		ptr = _DataConstraints.ptr<float>(_DCc); 
		ptr[0] = _rowCount; ptr[1] = _x_index[i*_width+j-1]; ptr[2]= v10; _DCc = _DCc+1;
		ptr = _DataConstraints.ptr<float>(_DCc); 
		ptr[0] = _rowCount; ptr[1] = _x_index[i*_width+j]; ptr[2]= v11; _DCc = _DCc+1;		
		
		b.at<float>(_rowCount,0) = _dataterm_element_desPt[k].x;
		_rowCount = _rowCount+1;

		ptr = _DataConstraints.ptr<float>(_DCc);       
		ptr[0] = _rowCount; ptr[1] = _y_index[(i-1)*_width+j-1]; ptr[2]= v00; _DCc = _DCc+1;
		ptr = _DataConstraints.ptr<float>(_DCc); 
		ptr[0] = _rowCount; ptr[1] = _y_index[(i-1)*_width+j]; ptr[2]= v01; _DCc = _DCc+1;
		ptr = _DataConstraints.ptr<float>(_DCc); 
		ptr[0] = _rowCount; ptr[1] = _y_index[i*_width+j-1]; ptr[2]= v10; _DCc = _DCc+1;
		ptr = _DataConstraints.ptr<float>(_DCc); 
		ptr[0] = _rowCount; ptr[1] = _y_index[i*_width+j]; ptr[2]= v11; _DCc = _DCc+1;		
		
		b.at<float>(_rowCount,0) = _dataterm_element_desPt[k].y;
		_rowCount = _rowCount+1;        


	}
}
void HomoTrans(double* ptr, float x, float y, float& wx, float& wy)
{
	wx = ptr[0]*x + ptr[1]*y + ptr[2];
	wy = ptr[3]*x + ptr[4]*y + ptr[5];
	float wz = ptr[6]*x + ptr[7]*y + ptr[8];
	wx /= wz;
	wy /= wz;
}

void ASAPWarping::AddDataCons(int i, int j, double* homoptr, cv::Mat& b)
{
	float wx(0),wy(0);
	int idx = i*_width + j;
	cv::Point2f pt = _source->getVertex(i,j);
	float x0 = pt.x;
	float y0 = pt.y;
	float x = _x_index[idx];
	float y = _y_index[idx];
	HomoTrans(homoptr,x0,y0,wx,wy);
	float * ptr = _DataConstraints.ptr<float>(_DCc);       
	ptr[0] = _rowCount; ptr[1] = x;	ptr[2] = 1;
	_DCc++;
	b.at<float>(_rowCount,0) = wx;
	_rowCount++;

	ptr = _DataConstraints.ptr<float>(_DCc);       
	ptr[0] = _rowCount; ptr[1] = y;	ptr[2] = 1;
	_DCc++;
	b.at<float>(_rowCount,0) = wy;
	_rowCount++;
}
void ASAPWarping::CreateMyDataCons(int num, std::vector<cv::Mat>& homographies, cv::Mat& b)
{
	int len = num;
	_num_data_cons = len*2*4;
	_DataConstraints.create(_num_data_cons,3,CV_32F);
	b.create(_num_data_cons+_num_smooth_cons,1,CV_32F);
	b = cv::Scalar(0);
	_DCc = 0;
	float * ptr = _DataConstraints.ptr<float>(_DCc);       
	
	for(int i=0; i<_height-1; i++)
	{
		for(int j=0; j< _width-1; j++)
		{
			if (!homographies[i*_quadStep + j].empty())
			{
				double* hptr = (double*)homographies[i*_quadStep + j].data;
				AddDataCons(i,j,hptr,b);
				AddDataCons(i,j+1,hptr,b);
				AddDataCons(i+1,j,hptr,b);
				AddDataCons(i+1,j+1,hptr,b);
			}
		}
	}
}
void ASAPWarping::CreateMyDataConsB(int num, std::vector<cv::Mat>& homographies, cv::Mat& b)
{
	//homo dataterm
	int len = num;
	_num_data_cons = len*2*4;
	//bileaner dataterm
	len = _dataterm_element_i.size();
	_num_data_cons += len*8;

	_DataConstraints.create(_num_data_cons,3,CV_32F);
	b.create(_num_data_cons+_num_smooth_cons,1,CV_32F);
	b = cv::Scalar(0);
	_DCc = 0;
	float * ptr = _DataConstraints.ptr<float>(_DCc);       
	
	for(int i=0; i<_height-1; i++)
	{
		for(int j=0; j< _width-1; j++)
		{
			if (!homographies[i*_quadStep + j].empty())
			{
				double* hptr = (double*)homographies[i*_quadStep + j].data;
				AddDataCons(i,j,hptr,b);
				AddDataCons(i,j+1,hptr,b);
				AddDataCons(i+1,j,hptr,b);
				AddDataCons(i+1,j+1,hptr,b);
			}
		}
	}

	for(int k=0; k<len; k++)
	{
		//std::cout<<k<<std::endl;
		float i = _dataterm_element_i[k];
		float j = _dataterm_element_j[k];
		float v00 = _dataterm_element_V00[k];
		float v01 = _dataterm_element_V01[k];
		float v10 = _dataterm_element_V10[k];
		float v11 = _dataterm_element_V11[k];
		float * ptr = _DataConstraints.ptr<float>(_DCc);       
		ptr[0] = _rowCount; ptr[1] = _x_index[(i-1)*_width+j-1]; ptr[2]= v00; _DCc = _DCc+1;
		ptr = _DataConstraints.ptr<float>(_DCc); 
		ptr[0] = _rowCount; ptr[1] = _x_index[(i-1)*_width+j]; ptr[2]= v01; _DCc = _DCc+1;
		ptr = _DataConstraints.ptr<float>(_DCc); 
		ptr[0] = _rowCount; ptr[1] = _x_index[i*_width+j-1]; ptr[2]= v10; _DCc = _DCc+1;
		ptr = _DataConstraints.ptr<float>(_DCc); 
		ptr[0] = _rowCount; ptr[1] = _x_index[i*_width+j]; ptr[2]= v11; _DCc = _DCc+1;		
		
		b.at<float>(_rowCount,0) = _dataterm_element_desPt[k].x;
		_rowCount = _rowCount+1;

		ptr = _DataConstraints.ptr<float>(_DCc);       
		ptr[0] = _rowCount; ptr[1] = _y_index[(i-1)*_width+j-1]; ptr[2]= v00; _DCc = _DCc+1;
		ptr = _DataConstraints.ptr<float>(_DCc); 
		ptr[0] = _rowCount; ptr[1] = _y_index[(i-1)*_width+j]; ptr[2]= v01; _DCc = _DCc+1;
		ptr = _DataConstraints.ptr<float>(_DCc); 
		ptr[0] = _rowCount; ptr[1] = _y_index[i*_width+j-1]; ptr[2]= v10; _DCc = _DCc+1;
		ptr = _DataConstraints.ptr<float>(_DCc); 
		ptr[0] = _rowCount; ptr[1] = _y_index[i*_width+j]; ptr[2]= v11; _DCc = _DCc+1;		
		
		b.at<float>(_rowCount,0) = _dataterm_element_desPt[k].y;
		_rowCount = _rowCount+1;        


	}

}
void ASAPWarping::Solve()
{
	cv::Mat b;
	CreateDataCons(b);
	int N = _SmoothConstraints.rows + _DataConstraints.rows;

	cv::Mat AMat(N,3,CV_32F);
	/*std::ofstream sc("smooth.txt");
	sc<<_SmoothConstraints;
	sc.close();
	std::ofstream dc("data.txt");
	dc<<_DataConstraints;
	dc.close();*/
	_SmoothConstraints.copyTo(AMat(cv::Rect(0,0,3,_SmoothConstraints.rows)));
	_DataConstraints.copyTo(AMat(cv::Rect(0,_SmoothConstraints.rows,3,_DataConstraints.rows)));

	//std::vector<double> bm(b.rows),x;
	//std::ofstream bfile("b.txt");
	//for(int i=0; i<bm.size(); i++)
	//{
	//	bm[i] = b.at<float>(i,0);
	//	//bfile<<bm[i]<<std::endl;
	//}
	/*bfile.close();*/
	/*b.col(0).copyTo(bm);
	SolveSparse(AMat,bm,x);*/
	
	std::vector<float> x;
	//CvLeastSquareSolve(b.rows,_columns,AMat,b,x);
	LeastSquareSolve(b.rows,_columns,AMat,b,x);
	int hwidth = _columns/2;
	for(int i=0; i<_height; i++)
	{
		for(int j=0; j<_width; j++)
		{
			cv::Point2f pt(x[i*_width+j],x[hwidth+i*_width+j]);
			_destin->setVertex(i,j,pt);
		}
	}
}
void ASAPWarping::MySolve(cv::Mat& b)
{
	int N = _SmoothConstraints.rows + _DataConstraints.rows;

	cv::Mat AMat(N,3,CV_32F);
	
	_SmoothConstraints.copyTo(AMat(cv::Rect(0,0,3,_SmoothConstraints.rows)));
	_DataConstraints.copyTo(AMat(cv::Rect(0,_SmoothConstraints.rows,3,_DataConstraints.rows)));

	
	
	std::vector<float> x;
	
	LeastSquareSolve(b.rows,_columns,AMat,b,x);
	int hwidth = _columns/2;
	for(int i=0; i<_height; i++)
	{
		for(int j=0; j<_width; j++)
		{
			cv::Point2f pt(x[i*_width+j],x[hwidth+i*_width+j]);
			_destin->setVertex(i,j,pt);
		}
	}
}
void ASAPWarping::GpuWarp(const cv::gpu::GpuMat& img, cv::gpu::GpuMat& warpImg, int gap)
{
	
	for(int i=1; i<_height; i++)
	{
		for(int j=1; j<_width; j++)
		{
			
			Quad qd1 = _source->getQuad(i,j);
			Quad qd2 = _destin->getQuad(i,j);
			calcQuadHomography(i-1,j-1,qd1,qd2);
		}
	}
	cudaMemcpy(_dBlkInvHomoVec,&_blkInvHomoVec[0],sizeof(double)*8*_blkSize,cudaMemcpyHostToDevice);
	cudaMemcpy(_dBlkHomoVec,&_blkHomoVec[0],sizeof(double)*8*_blkSize,cudaMemcpyHostToDevice);

	CudaWarp(img, _quadStep,_dBlkHomoVec,_dBlkInvHomoVec,_dMapXY[0],_dMapXY[1], _dIMapXY[0],_dIMapXY[1],warpImg);

	cv::gpu::merge(_dMapXY,2,_dMap);
	cv::gpu::merge(_dIMapXY,2,_dIMap);
	_dMap.download(_mapXY);
	_dMapXY[0].download(_maps[0]);
	_dMapXY[1].download(_maps[1]);
	//_dIMap.download(_invMapXY);
	_dIMapXY[0].download(_invMaps[0]);
	_dIMapXY[1].download(_invMaps[1]);
}
void ASAPWarping::Warp(const cv::Mat& img1, cv::Mat& warpImg, int gap)
{
	_warpImg = cv::Mat::zeros(img1.rows+2*gap,img1.cols+2*gap,CV_8UC3);
	for(int i=1; i<_height; i++)
	{
		for(int j=1; j<_width; j++)
		{
			
			Quad qd1 = _source->getQuad(i,j);
			Quad qd2 = _destin->getQuad(i,j);
			quadWarp(img1,i-1,j-1,qd1,qd2);
		}
	}
	//warpImg = _warpImg.clone();
	cv::remap(img1,warpImg,_maps[0],_maps[1],CV_INTER_CUBIC);
	

	cv::merge(_maps,2,_mapXY);
	//std::cout<<_mapXY;
	_dMap.upload(_mapXY);
	
	cv::merge(_invMaps,2,_invMapXY);
	_dIMap.upload(_invMapXY);
	_dIMapXY[0].upload(_invMaps[0]);
	_dIMapXY[1].upload(_invMaps[1]);

}
void ASAPWarping::calcQuadHomography(int row, int col, Quad& q1, Quad& q2)
{
	float minx = q2.getMinX();
    float maxx = q2.getMaxX();
    float  miny = q2.getMinY();
    float  maxy = q2.getMaxY();
             
    std::vector<cv::Point2f> f1,f2;
	f1.push_back(q1.getV00());
	f1.push_back(q1.getV01());
	f1.push_back(q1.getV10());
	f1.push_back(q1.getV11());
             
	f2.push_back(q2.getV00());
	f2.push_back(q2.getV01());
	f2.push_back(q2.getV10());
	f2.push_back(q2.getV11());
	
	cv::Mat homography;	
	findHomographySVD(f2,f1,homography);
	cv::Mat invHomo = homography.inv();
	int idx = row*(_width-1)+col;
	memcpy(&_blkHomoVec[idx*8],homography.data,64);
	memcpy(&_blkInvHomoVec[idx*8],invHomo.data,64);
	
}
void ASAPWarping::quadWarp(const cv::Mat& img, int row, int col, Quad& q1, Quad& q2)
{
	float minx = q2.getMinX();
    float maxx = q2.getMaxX();
    float  miny = q2.getMinY();
    float  maxy = q2.getMaxY();
             
    std::vector<cv::Point2f> f1,f2;
	f1.push_back(q1.getV00());
	f1.push_back(q1.getV01());
	f1.push_back(q1.getV10());
	f1.push_back(q1.getV11());
             
	f2.push_back(q2.getV00());
	f2.push_back(q2.getV01());
	f2.push_back(q2.getV10());
	f2.push_back(q2.getV11());
	
	cv::Mat homography;	
	findHomographySVD(f1,f2,homography);
	cv::Mat invHomo = homography.inv();
	_homographies[row*(_width-1)+col] = homography.clone();
	_invHomographies[row*(_width-1)+col] =invHomo.clone();
	//std::cout<<homography;
	
             //qd = Quad(q2.V00,q2.V01,q2.V10,q2.V11);
           // _warpIm = myWarp(minx,maxx,miny,maxy,im,obj.warpIm,H,obj.gap);
            //_warpIm = uint8(obj.warpIm);
	/*if(gap > 0)
	{
		minx = floor(minx);
		miny =floor(miny);
	}
	else
	{
		minx = max(floor(minx),1);
		miny = max(floor(miny),1);
	}*/
	int width = img.cols;
	int height = img.rows;
	/*maxx = min(ceil(maxx),w);
	maxy = min(ceil(maxy),h);*/
	/*cv::Size size = cv::Size(maxx-minx,maxy-miny);
	cv::Mat map_x,map_y;*/
	
	double* ptr = (double*)invHomo.data;
	double* invPtr = (double*)homography.data;
	for(int i=0; i<_quadHeight; i++)
	{
		int r = row*_quadHeight+i;
		float* ptrX = _maps[0].ptr<float>(r);
		float* ptrY = _maps[1].ptr<float>(r);
		float* invPtrX = _invMaps[0].ptr<float>(r);
		float* invPtrY = _invMaps[1].ptr<float>(r);
		for(int j=0; j<_quadWidth; j++)
		{
			int c = col*_quadWidth+j;
			float x,y,w;
			x = c*ptr[0] + r*ptr[1] + ptr[2];
			y = c*ptr[3] + r*ptr[4] + ptr[5];
			w = c*ptr[6] + r*ptr[7] + ptr[8];
			x /=w;
			y/=w;
			ptrX[c] = x;
			ptrY[c] = y;

			x = c*invPtr[0] + r*invPtr[1] + invPtr[2];
			y = c*invPtr[3] + r*invPtr[4] + invPtr[5];
			w = c*invPtr[6] + r*invPtr[7] + invPtr[8];
			x /=w;
			y/=w;
			invPtrX[c] = x;
			invPtrY[c] = y;
			
		}
	}


}
void ASAPWarping::getFlow(cv::Mat& flow)
{
	flow.create(_maps[0].size(),CV_32FC2);
	for(int i=0; i<flow.rows; i++)
	{
		float* ptrX = _maps[0].ptr<float>(i);
		float* ptrY = _maps[1].ptr<float>(i);
		cv::Vec2f* ptrFlow = flow.ptr<cv::Vec2f>(i);
		for(int j=0; j<flow.cols; j++)
		{
			ptrFlow[j] = cv::Vec2f(j-ptrX[j],i-ptrY[j]);
		}
	}
}

