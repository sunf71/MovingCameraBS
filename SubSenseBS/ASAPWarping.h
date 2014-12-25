#pragma once
#include <vector>
#include <math.h>
#include <opencv2\opencv.hpp>
using namespace std;
bool isPointInTriangular(const cv::Point2f& pt, const cv::Point2f& V0, const cv::Point2f& V1, const cv::Point2f& V2);




class Quad
{
public:
	Quad(const cv::Point2f& v00, const cv::Point2f& v01, const cv::Point2f& v10, const cv::Point2f& v11):_V00(v00),_V01(v01),_V10(v10),_V11(v11){};
	bool isPointIn(const cv::Point2f& pt)
	{
		bool in1 = isPointInTriangular(pt,_V00,_V01,_V11);
		bool in2 = isPointInTriangular(pt,_V00,_V10,_V11);
		return in1 || in2;
	}
	bool getBilinearCoordinates(const cv::Point2f& pt, std::vector<float>& coefficients)
	{
		coefficients.clear();
		float a_x = _V00.x - _V01.x - _V10.x + _V11.x;
		float  b_x = -_V00.x + _V01.x;
		float c_x = -_V00.x + _V10.x;
		float d_x = _V00.x - pt.x;

		float  a_y = _V00.y - _V01.y - _V10.y + _V11.y;
		float b_y = -_V00.y + _V01.y;
		float c_y = -_V00.y + _V10.y;
		float d_y = _V00.y - pt.y;

		float bigA = -a_y*b_x + b_y*a_x;
		float bigB = -a_y*d_x - c_y*b_x + d_y*a_x +b_y*c_x;
		float  bigC = -c_y*d_x + d_y*c_x;

		float tmp1 = -1;
		float tmp2 = -1;
		float tmp3 = -1;
		float tmp4 = -1;
		float k1,k2;
		if (bigB*bigB - 4*bigA*bigC >= 0.0)
		{
			if( abs(bigA) >= 0.000001)
			{
				tmp1 = ( -bigB + sqrt(bigB*bigB - 4*bigA*bigC) ) / ( 2*bigA );
				tmp2 = ( -bigB - sqrt(bigB*bigB - 4*bigA*bigC) ) / ( 2*bigA );
			}
			else                   
			{
				tmp1 = -bigC/bigB;
			}


			if ( tmp1 >= -0.999999 && tmp1 <= 1.000001)
			{   
				tmp3 = -(b_y*tmp1 + d_y) / (a_y*tmp1 + c_y);
				tmp4 = -(b_x*tmp1 + d_x) / (a_x*tmp1 + c_x);
				if (tmp3 >= -0.999999 && tmp3 <= 1.000001)
				{   
					k1 = tmp1;

					k2 = tmp3;
				}
				else if (tmp4 >= -0.999999 && tmp4 <= 1.000001)
				{
					k1 = tmp1;
					k2 = tmp4;
				}
			}
		}
		if ( tmp2 >= -0.999999 && tmp2 <= 1.000001)
		{   
			if (tmp3 >= -0.999999 && tmp3 <= 1.000001)
			{
				k1 = tmp2;
				k2 = tmp3;
			}
			else if (tmp4 >= -0.999999 && tmp4 <= 1.000001)
			{
				k1 = tmp2;
				k2 = tmp4;
			}
		}


		if (k1>=-0.999999 && k1<=1.000001 && k2>=-0.999999 && k2<=1.000001)
		{

			coefficients.push_back((1.0-k1)*(1.0-k2));
			coefficients.push_back(k1*(1.0-k2));
			coefficients.push_back((1.0-k1)*k2);
			coefficients.push_back(k1*k2);



			return true;
		}
		else
		{
			return false;
		}

	}
	float getMinX()
	{

		float  minx = std::min(_V00.x,_V01.x);
		minx = min(minx, _V10.x);
		minx = min(minx,_V11.x);
		return minx;
	}
	float getMaxX()
	{
		float  maxx = max(_V00.x,_V01.x);
		maxx = max(maxx, _V10.x);
		maxx = max(maxx,_V11.x);
		return maxx;
	}

	float  getMinY()
	{     
		float miny = min(_V00.y,_V01.y);
		miny = min(miny,_V10.y);
		miny = min(miny,_V11.y);
		return miny;
	}
	float getMaxY()
	{
		float maxy = max(_V00.y,_V01.y);
		maxy = max(maxy,_V10.y);
		maxy = max(maxy,_V11.y);
		return maxy;
	}
private:

	cv::Point2f _V00,_V01,_V10,_V11;
};

class Mesh
{
public:
	Mesh(int rows, int cols, int quadWidth, int quadHeight):_imgWidth(cols), _imgHeight(rows),_quadWidth(quadWidth), _quadHeight(quadHeight)
	{
		std::vector<int> quadX, quadY;
	
		int x = 0;
		int halfWidth = _quadWidth/2;
		while( _imgWidth-1 - x > halfWidth)
		{
			quadX.push_back(x);
			x += _quadWidth;
		}
		quadX.push_back(_imgWidth-1);
		int halfHeight = _quadHeight/2;
		int y = 0;
		while(_imgHeight - y -1> halfHeight)
		{
			quadY.push_back(y);
			y += _quadHeight;
		}
		quadY.push_back(_imgHeight-1);

		_meshWidth = quadX.size();
		_meshHeight = quadY.size();

		_xMat.create(_meshHeight,_meshWidth,CV_32F);
		_yMat.create(_meshHeight,_meshWidth,CV_32F);

		for(int i=0; i< _meshHeight; i++)
		{
			float* xPtr = _xMat.ptr<float>(i);
			float* yPtr = _yMat.ptr<float>(i);
			for(int j=0; j<_meshWidth; j++)
			{
				xPtr[j] = quadX[j];
				yPtr[j] = quadY[i];
			}
		}
	};
public:
	cv::Point2f getVertex(int i, int j)
	{
		if ( i<_meshWidth && j < _meshHeight)
			return cv::Point2f(_xMat.at<float>(i,j), _yMat.at<float>(i,j));
		else
			return cv::Point2f();
	}
	void setVertex(int i, int j, cv::Point2f& pt)
	{
		_xMat.at<float>(i,j) = pt.x;
		_yMat.at<float>(i,j) = pt.y;
	}

	Quad getQuad(int i, int j)
	{
		cv::Point2f v00 = getVertex(i,j);
		cv::Point2f v01 = getVertex(i,j+1);
		cv::Point2f v10 = getVertex(i+1,j);
		cv::Point2f v11 = getVertex(i+1,j+1);
		return Quad(v00,v01,v10,v11);
	}
	void drawMesh(const cv::Mat& img, int gap, cv::Mat& nimg)
	{
		nimg.create(img.rows+2*gap,img.cols+2*gap,img.type());		
		img.copyTo(nimg(cv::Rect(gap,gap,img.cols,img.rows)));
		cv::Scalar color(255,0,0);
		cv::Scalar color2(0,255,0);
		for(int i=0; i<_meshHeight; i++)
		{
			for(int j=0; j<_meshWidth; j++)
			{
				cv::Point2f point;
				point.x = _xMat.at<float>(i,j)+gap;
				point.y = _yMat.at<float>(i,j) +gap;
				cv::circle(nimg, point,3,color);
			}
		}
		for(int i=0; i<_meshHeight; i++)
		{
			cv::Point2f start,end;
			start.x = _xMat.at<float>(i,0)+gap;
			start.y = _yMat.at<float>(i,0)+gap;
			end.x = _xMat.at<float>(i,_meshWidth-1) + gap;
			end.y = _yMat.at<float>(i,_meshWidth-1) + gap;
			cv::line(nimg,start,end,color2);

		}
		for(int i=0; i<_meshWidth; i++)
		{
			cv::Point2f start,end;
			start.x = _xMat.at<float>(0,i)+gap;
			start.y = _yMat.at<float>(0,i)+gap;
			end.x = _xMat.at<float>(_meshHeight-1,i) + gap;
			end.y = _yMat.at<float>(_meshHeight-1,i) + gap;
			cv::line(nimg,start,end,color2);
		}
	}


	int _imgWidth, _imgHeight, _meshWidth, _meshHeight, _quadWidth, _quadHeight;

	cv::Mat _xMat, _yMat;
};

class ASAPWarping
{
public:
	ASAPWarping(int height, int width, int quadWidth, int quadHeight, float weight):_quadWidth(quadWidth),_quadHeight(quadHeight),_weight(weight)
	{
		_source = new Mesh(height,width,quadWidth,quadHeight);
		_destin = new Mesh(height,width,quadWidth,quadHeight);
		_height = _source->_meshHeight;
		_width = _source->_meshWidth;

		_x_index.resize(_height*_width);
		_y_index.resize(_height*_width);
		for(int i=0; i<_height*_width; i++)
		{
			_x_index[i] = i;
			_y_index[i] = _height*_width+i;
		}
		_num_smooth_cons = (_height-2)*(_width-2)*16 + (2*(_width+_height)-8)*8+4*4;

		_columns = _width*_height*2;

		_SmoothConstraints = cv::Mat::zeros(_num_smooth_cons*5,3,CV_32F);
		_SCc = 0;

		CreateSmoothCons(weight);
		//std::cout<<_SmoothConstraints;
	};
	~ASAPWarping()
	{
		delete _source;
		delete _destin;
	}
	void SetControlPts(std::vector<cv::Point2f>& inputsPts, std::vector<cv::Point2f>& outputsPts);

	void CreateDataCons(cv::Mat& b);
protected:
	void getSmoothWeight(const cv::Point2f& V1, const cv::Point2f& V2, const cv::Point2f& V3, float& u, float& v)
	{
		float d1 = sqrt((V1.x - V2.x)*(V1.x - V2.x) + (V1.y - V2.y)*(V1.y - V2.y));
		float   d3 = sqrt((V2.x - V3.x)*(V2.x - V3.x) + (V2.y - V3.y)*(V2.y - V3.y));

		cv::Point2f v21 = cv::Point2f(V1.x-V2.x,V1.y-V2.y);
		cv::Point2f v23 = cv::Point2f(V3.x-V2.x,V3.y-V2.y);

		float cosin = v21.x*v23.x + v21.y*v23.y;
		cosin = cosin/(d1*d3);

		float u_dis = cosin*d1;
		u = u_dis/d3;

		float v_dis = sqrt(d1*d1 - u_dis*u_dis);
		v = v_dis/d3;
	}

	void addCoefficient_1(int i, int j, float weight)
	{
		//V3(i-1,j-1)
		//|
		//|____
		//V2   V1(i,j)
		cv::Point2f V1 = _source->getVertex(i,j);
		cv::Point2f  V2 = _source->getVertex(i,j-1);
		cv::Point2f  V3 = _source->getVertex(i-1,j-1);

		float u,v;
		getSmoothWeight(V1,V2,V3,u,v);

		int coordv1 =  i*_width+j;
		int coordv2 =  i*_width+j-1;
		int coordv3 = (i-1)*_width+j-1;            


		//            _SmoothCons.add_row(rowCount,_x_index(coordv2),(1.0-u)*weight);
		//            _SmoothCons.add_row(rowCount,_x_index(coordv3),u*weight);
		//            _SmoothCons.add_row(rowCount,_y_index(coordv2),v*weight);
		//            _SmoothCons.add_row(rowCount,_y_index(coordv3),(-1.0*v)*weight);
		//            _SmoothCons.add_row(rowCount,_x_index(coordv1),(-1.0)*weight);

		//V1.x =  V2.x + u * (V3.x - V2.x) + v * (V2.y - V3.y);
		float* ptr = _SmoothConstraints.ptr<float>(_SCc);
		ptr[0] = _rowCount; ptr[1] = _x_index[coordv2]; ptr[2] = (1.0-u)*weight;  _SCc = _SCc + 1;
		ptr = _SmoothConstraints.ptr<float>(_SCc);
		ptr[0] = _rowCount; ptr[1] = _x_index[coordv3]; ptr[2] = u*weight;  _SCc = _SCc + 1;
		ptr = _SmoothConstraints.ptr<float>(_SCc);
		ptr[0] = _rowCount; ptr[1] = _y_index[coordv2]; ptr[2] = v*weight;  _SCc = _SCc + 1;
		ptr = _SmoothConstraints.ptr<float>(_SCc);
		ptr[0] = _rowCount; ptr[1] = _y_index[coordv3]; ptr[2] = -1.0*v*weight;  _SCc = _SCc + 1;
		ptr = _SmoothConstraints.ptr<float>(_SCc);
		ptr[0] = _rowCount; ptr[1] = _x_index[coordv1]; ptr[2] = -1.0*weight;  _SCc = _SCc + 1;

		_rowCount = _rowCount+1; 


		//            _SmoothCons.add_row(rowCount,_y_index(coordv2),(1.0-u)*weight);
		//            _SmoothCons.add_row(rowCount,_y_index(coordv3),u*weight);
		//            _SmoothCons.add_row(rowCount,_x_index(coordv3),v*weight);
		//            _SmoothCons.add_row(rowCount,_x_index(coordv2),(-1.0*v)*weight);
		//            _SmoothCons.add_row(rowCount,_y_index(coordv1),(-1.0)*weight);
		//            rowCount = rowCount+1;

		//V1.y = V2.y + u * (V3.y - V2.y) + v * (V3.x - V2.x);
		ptr = _SmoothConstraints.ptr<float>(_SCc);
		ptr[0] = _rowCount; ptr[1] = _y_index[coordv2]; ptr[2] = (1.0-u)*weight;  _SCc = _SCc + 1;
		ptr = _SmoothConstraints.ptr<float>(_SCc);
		ptr[0] = _rowCount; ptr[1] = _y_index[coordv3]; ptr[2] = u*weight;  _SCc = _SCc + 1;
		ptr = _SmoothConstraints.ptr<float>(_SCc);
		ptr[0] = _rowCount; ptr[1] = _x_index[coordv3]; ptr[2] = v*weight;  _SCc = _SCc + 1;
		ptr = _SmoothConstraints.ptr<float>(_SCc);
		ptr[0] = _rowCount; ptr[1] = _x_index[coordv2]; ptr[2] = -1.0*v*weight;  _SCc = _SCc + 1;
		ptr = _SmoothConstraints.ptr<float>(_SCc);
		ptr[0] = _rowCount; ptr[1] = _y_index[coordv1]; ptr[2] = -1.0*weight;  _SCc = _SCc + 1;

		_rowCount = _rowCount+1; 



	}
	void addCoefficient_2(int i, int j, float weight)
	{
		//     V3   V2
		//       _____
		//       |
		//       |
		//      V1(i,j)
		cv::Point2f V1 = _source->getVertex(i,j);
		cv::Point2f  V2 = _source->getVertex(i-1,j);
		cv::Point2f  V3 = _source->getVertex(i-1,j-1);

		float u,v;
		getSmoothWeight(V1,V2,V3,u,v);

		int coordv1 =  i*_width+j;
		int coordv2 =  (i-1)*_width+j;
		int coordv3 = (i-1)*_width+j-1;            


		//            _SmoothCons.add_row(rowCount,_x_index(coordv2),(1.0-u)*weight);
		//            _SmoothCons.add_row(rowCount,_x_index(coordv3),u*weight);
		//            _SmoothCons.add_row(rowCount,_y_index(coordv3),v*weight);
		//            _SmoothCons.add_row(rowCount,_y_index(coordv2),(-1.0*v)*weight);
		//            _SmoothCons.add_row(rowCount,_x_index(coordv1),(-1.0)*weight);

		//V1.x =  V2.x + u * (V3.x - V2.x) + v * (V2.y - V3.y);
		float* ptr = _SmoothConstraints.ptr<float>(_SCc);
		ptr[0] = _rowCount; ptr[1] = _x_index[coordv2]; ptr[2] = (1.0-u)*weight;  _SCc = _SCc + 1;
		ptr = _SmoothConstraints.ptr<float>(_SCc);
		ptr[0] = _rowCount; ptr[1] = _x_index[coordv3]; ptr[2] = u*weight;  _SCc = _SCc + 1;
		ptr = _SmoothConstraints.ptr<float>(_SCc);
		ptr[0] = _rowCount; ptr[1] = _y_index[coordv3]; ptr[2] = v*weight;  _SCc = _SCc + 1;
		ptr = _SmoothConstraints.ptr<float>(_SCc);
		ptr[0] = _rowCount; ptr[1] = _y_index[coordv2]; ptr[2] = -1.0*v*weight;  _SCc = _SCc + 1;
		ptr = _SmoothConstraints.ptr<float>(_SCc);
		ptr[0] = _rowCount; ptr[1] = _x_index[coordv1]; ptr[2] = -1.0*weight;  _SCc = _SCc + 1;

		_rowCount = _rowCount+1; 


		//            _SmoothCons.add_row(rowCount,_y_index(coordv2),(1.0-u)*weight);
		//            _SmoothCons.add_row(rowCount,_y_index(coordv3),u*weight);
		//            _SmoothCons.add_row(rowCount,_x_index(coordv2),v*weight);
		//            _SmoothCons.add_row(rowCount,_x_index(coordv3),(-1.0*v)*weight);
		//            _SmoothCons.add_row(rowCount,_y_index(coordv1),(-1.0)*weight);
		//            rowCount = rowCount+1;

		//V1.y = V2.y + u * (V3.y - V2.y) + v * (V3.x - V2.x);
		ptr = _SmoothConstraints.ptr<float>(_SCc);
		ptr[0] = _rowCount; ptr[1] = _y_index[coordv2]; ptr[2] = (1.0-u)*weight;  _SCc = _SCc + 1;
		ptr = _SmoothConstraints.ptr<float>(_SCc);
		ptr[0] = _rowCount; ptr[1] = _y_index[coordv3]; ptr[2] = u*weight;  _SCc = _SCc + 1;
		ptr = _SmoothConstraints.ptr<float>(_SCc);
		ptr[0] = _rowCount; ptr[1] = _x_index[coordv2]; ptr[2] = v*weight;  _SCc = _SCc + 1;
		ptr = _SmoothConstraints.ptr<float>(_SCc);
		ptr[0] = _rowCount; ptr[1] = _x_index[coordv3]; ptr[2] = -1.0*v*weight;  _SCc = _SCc + 1;
		ptr = _SmoothConstraints.ptr<float>(_SCc);
		ptr[0] = _rowCount; ptr[1] = _y_index[coordv1]; ptr[2] = -1.0*weight;  _SCc = _SCc + 1;

		_rowCount = _rowCount+1; 



	}
	void addCoefficient_3(int i, int j, float weight)
	{
		//V3(i-1,j-1)
		//|
		//|____
		//V2   V1(i,j)
		cv::Point2f V1 = _source->getVertex(i,j);
		cv::Point2f  V2 = _source->getVertex(i-1,j);
		cv::Point2f  V3 = _source->getVertex(i-1,j+1);

		float u,v;
		getSmoothWeight(V1,V2,V3,u,v);

		int coordv1 =  i*_width+j;
		int coordv2 =  (i-1)*_width+j;
		int coordv3 = (i-1)*_width+j+1;            


		//            _SmoothCons.add_row(rowCount,_x_index(coordv2),(1.0-u)*weight);
		//            _SmoothCons.add_row(rowCount,_x_index(coordv3),u*weight);
		//            _SmoothCons.add_row(rowCount,_y_index(coordv2),v*weight);
		//            _SmoothCons.add_row(rowCount,_y_index(coordv3),(-1.0*v)*weight);
		//            _SmoothCons.add_row(rowCount,_x_index(coordv1),(-1.0)*weight);

		//V1.x =  V2.x + u * (V3.x - V2.x) + v * (V2.y - V3.y);
		float* ptr = _SmoothConstraints.ptr<float>(_SCc);
		ptr[0] = _rowCount; ptr[1] = _x_index[coordv2]; ptr[2] = (1.0-u)*weight;  _SCc = _SCc + 1;
		ptr = _SmoothConstraints.ptr<float>(_SCc);
		ptr[0] = _rowCount; ptr[1] = _x_index[coordv3]; ptr[2] = u*weight;  _SCc = _SCc + 1;
		ptr = _SmoothConstraints.ptr<float>(_SCc);
		ptr[0] = _rowCount; ptr[1] = _y_index[coordv2]; ptr[2] = v*weight;  _SCc = _SCc + 1;
		ptr = _SmoothConstraints.ptr<float>(_SCc);
		ptr[0] = _rowCount; ptr[1] = _y_index[coordv3]; ptr[2] = -1.0*v*weight;  _SCc = _SCc + 1;
		ptr = _SmoothConstraints.ptr<float>(_SCc);
		ptr[0] = _rowCount; ptr[1] = _x_index[coordv1]; ptr[2] = -1.0*weight;  _SCc = _SCc + 1;

		_rowCount = _rowCount+1; 


		//            _SmoothCons.add_row(rowCount,_y_index(coordv2),(1.0-u)*weight);
		//            _SmoothCons.add_row(rowCount,_y_index(coordv3),u*weight);
		//            _SmoothCons.add_row(rowCount,_x_index(coordv3),v*weight);
		//            _SmoothCons.add_row(rowCount,_x_index(coordv2),(-1.0*v)*weight);
		//            _SmoothCons.add_row(rowCount,_y_index(coordv1),(-1.0)*weight);
		//            rowCount = rowCount+1;

		//V1.y = V2.y + u * (V3.y - V2.y) + v * (V3.x - V2.x);
		ptr = _SmoothConstraints.ptr<float>(_SCc);
		ptr[0] = _rowCount; ptr[1] = _y_index[coordv2]; ptr[2] = (1.0-u)*weight;  _SCc = _SCc + 1;
		ptr = _SmoothConstraints.ptr<float>(_SCc);
		ptr[0] = _rowCount; ptr[1] = _y_index[coordv3]; ptr[2] = u*weight;  _SCc = _SCc + 1;
		ptr = _SmoothConstraints.ptr<float>(_SCc);
		ptr[0] = _rowCount; ptr[1] = _x_index[coordv3]; ptr[2] = v*weight;  _SCc = _SCc + 1;
		ptr = _SmoothConstraints.ptr<float>(_SCc);
		ptr[0] = _rowCount; ptr[1] = _x_index[coordv2]; ptr[2] = -1.0*v*weight;  _SCc = _SCc + 1;
		ptr = _SmoothConstraints.ptr<float>(_SCc);
		ptr[0] = _rowCount; ptr[1] = _y_index[coordv1]; ptr[2] = -1.0*weight;  _SCc = _SCc + 1;

		_rowCount = _rowCount+1; 



	}
	void addCoefficient_4(int i, int j, float weight)
	{
		//          V3(i-1,j+1)
		//            |
		//    ______|
		//   V1     V2
		//   
		cv::Point2f V1 = _source->getVertex(i,j);
		cv::Point2f  V2 = _source->getVertex(i,j+1);
		cv::Point2f  V3 = _source->getVertex(i-1,j+1);

		float u,v;
		getSmoothWeight(V1,V2,V3,u,v);

		int coordv1 =  i*_width+j;
		int coordv2 =  i*_width+j+1;
		int coordv3 = (i-1)*_width+j+1;            


		//            _SmoothCons.add_row(rowCount,_x_index(coordv2),(1.0-u)*weight);
		//            _SmoothCons.add_row(rowCount,_x_index(coordv3),u*weight);
		//            _SmoothCons.add_row(rowCount,_y_index(coordv3),v*weight);
		//            _SmoothCons.add_row(rowCount,_y_index(coordv2),(-1.0*v)*weight);
		//            _SmoothCons.add_row(rowCount,_x_index(coordv1),(-1.0)*weight);

		//V1.x =  V2.x + u * (V3.x - V2.x) + v * (V2.y - V3.y);
		float* ptr = _SmoothConstraints.ptr<float>(_SCc);
		ptr[0] = _rowCount; ptr[1] = _x_index[coordv2]; ptr[2] = (1.0-u)*weight;  _SCc = _SCc + 1;
		ptr = _SmoothConstraints.ptr<float>(_SCc);
		ptr[0] = _rowCount; ptr[1] = _x_index[coordv3]; ptr[2] = u*weight;  _SCc = _SCc + 1;
		ptr = _SmoothConstraints.ptr<float>(_SCc);
		ptr[0] = _rowCount; ptr[1] = _y_index[coordv3]; ptr[2] = v*weight;  _SCc = _SCc + 1;
		ptr = _SmoothConstraints.ptr<float>(_SCc);
		ptr[0] = _rowCount; ptr[1] = _y_index[coordv2]; ptr[2] = -1.0*v*weight;  _SCc = _SCc + 1;
		ptr = _SmoothConstraints.ptr<float>(_SCc);
		ptr[0] = _rowCount; ptr[1] = _x_index[coordv1]; ptr[2] = -1.0*weight;  _SCc = _SCc + 1;

		_rowCount = _rowCount+1; 


		//            _SmoothCons.add_row(rowCount,_y_index(coordv2),(1.0-u)*weight);
		//            _SmoothCons.add_row(rowCount,_y_index(coordv3),u*weight);
		//            _SmoothCons.add_row(rowCount,_x_index(coordv2),v*weight);
		//            _SmoothCons.add_row(rowCount,_x_index(coordv3),(-1.0*v)*weight);
		//            _SmoothCons.add_row(rowCount,_y_index(coordv1),(-1.0)*weight);
		//            rowCount = rowCount+1;

		//V1.y = V2.y + u * (V3.y - V2.y) + v * (V3.x - V2.x);
		ptr = _SmoothConstraints.ptr<float>(_SCc);
		ptr[0] = _rowCount; ptr[1] = _y_index[coordv2]; ptr[2] = (1.0-u)*weight;  _SCc = _SCc + 1;
		ptr = _SmoothConstraints.ptr<float>(_SCc);
		ptr[0] = _rowCount; ptr[1] = _y_index[coordv3]; ptr[2] = u*weight;  _SCc = _SCc + 1;
		ptr = _SmoothConstraints.ptr<float>(_SCc);
		ptr[0] = _rowCount; ptr[1] = _x_index[coordv2]; ptr[2] = v*weight;  _SCc = _SCc + 1;
		ptr = _SmoothConstraints.ptr<float>(_SCc);
		ptr[0] = _rowCount; ptr[1] = _x_index[coordv3]; ptr[2] = -1.0*v*weight;  _SCc = _SCc + 1;
		ptr = _SmoothConstraints.ptr<float>(_SCc);
		ptr[0] = _rowCount; ptr[1] = _y_index[coordv1]; ptr[2] = -1.0*weight;  _SCc = _SCc + 1;

		_rowCount = _rowCount+1; 



	}
	void addCoefficient_5(int i, int j, float weight)
	{
		//             V1   V2
		//             _____
		//             |
		//             |
		//             V3
		cv::Point2f V1 = _source->getVertex(i,j);
		cv::Point2f  V2 = _source->getVertex(i,j+1);
		cv::Point2f  V3 = _source->getVertex(i+1,j+1);

		float u,v;
		getSmoothWeight(V1,V2,V3,u,v);

		int coordv1 =  i*_width+j;
		int coordv2 =  i*_width+j+1;
		int coordv3 = (i+1)*_width+j+1;            


		//            _SmoothCons.add_row(rowCount,_x_index(coordv2),(1.0-u)*weight);
		//            _SmoothCons.add_row(rowCount,_x_index(coordv3),u*weight);
		//            _SmoothCons.add_row(rowCount,_y_index(coordv2),v*weight);
		//            _SmoothCons.add_row(rowCount,_y_index(coordv3),(-1.0*v)*weight);
		//            _SmoothCons.add_row(rowCount,_x_index(coordv1),(-1.0)*weight);

		//V1.x =  V2.x + u * (V3.x - V2.x) + v * (V2.y - V3.y);
		float* ptr = _SmoothConstraints.ptr<float>(_SCc);
		ptr[0] = _rowCount; ptr[1] = _x_index[coordv2]; ptr[2] = (1.0-u)*weight;  _SCc = _SCc + 1;
		ptr = _SmoothConstraints.ptr<float>(_SCc);
		ptr[0] = _rowCount; ptr[1] = _x_index[coordv3]; ptr[2] = u*weight;  _SCc = _SCc + 1;
		ptr = _SmoothConstraints.ptr<float>(_SCc);
		ptr[0] = _rowCount; ptr[1] = _y_index[coordv2]; ptr[2] = v*weight;  _SCc = _SCc + 1;
		ptr = _SmoothConstraints.ptr<float>(_SCc);
		ptr[0] = _rowCount; ptr[1] = _y_index[coordv3]; ptr[2] = -1.0*v*weight;  _SCc = _SCc + 1;
		ptr = _SmoothConstraints.ptr<float>(_SCc);
		ptr[0] = _rowCount; ptr[1] = _x_index[coordv1]; ptr[2] = -1.0*weight;  _SCc = _SCc + 1;

		_rowCount = _rowCount+1; 


		//            _SmoothCons.add_row(rowCount,_y_index(coordv2),(1.0-u)*weight);
		//            _SmoothCons.add_row(rowCount,_y_index(coordv3),u*weight);
		//            _SmoothCons.add_row(rowCount,_x_index(coordv3),v*weight);
		//            _SmoothCons.add_row(rowCount,_x_index(coordv2),(-1.0*v)*weight);
		//            _SmoothCons.add_row(rowCount,_y_index(coordv1),(-1.0)*weight);
		//            rowCount = rowCount+1;

		//V1.y = V2.y + u * (V3.y - V2.y) + v * (V3.x - V2.x);
		ptr = _SmoothConstraints.ptr<float>(_SCc);
		ptr[0] = _rowCount; ptr[1] = _y_index[coordv2]; ptr[2] = (1.0-u)*weight;  _SCc = _SCc + 1;
		ptr = _SmoothConstraints.ptr<float>(_SCc);
		ptr[0] = _rowCount; ptr[1] = _y_index[coordv3]; ptr[2] = u*weight;  _SCc = _SCc + 1;
		ptr = _SmoothConstraints.ptr<float>(_SCc);
		ptr[0] = _rowCount; ptr[1] = _x_index[coordv3]; ptr[2] = v*weight;  _SCc = _SCc + 1;
		ptr = _SmoothConstraints.ptr<float>(_SCc);
		ptr[0] = _rowCount; ptr[1] = _x_index[coordv2]; ptr[2] = -1.0*v*weight;  _SCc = _SCc + 1;
		ptr = _SmoothConstraints.ptr<float>(_SCc);
		ptr[0] = _rowCount; ptr[1] = _y_index[coordv1]; ptr[2] = -1.0*weight;  _SCc = _SCc + 1;

		_rowCount = _rowCount+1; 



	}
	void addCoefficient_6(int i, int j, float weight)
	{
		//             V1(i,j)
		//             |
		//             |____
		//             V2   V3(i+1,j+1)
		//             (i+1,j)
		cv::Point2f V1 = _source->getVertex(i,j);
		cv::Point2f  V2 = _source->getVertex(i+1,j);
		cv::Point2f  V3 = _source->getVertex(i+1,j+1);

		float u,v;
		getSmoothWeight(V1,V2,V3,u,v);

		int coordv1 =  i*_width+j;
		int coordv2 =  (i+1)*_width+j;
		int coordv3 = (i+1)*_width+j+1;            


		//            _SmoothCons.add_row(rowCount,_x_index(coordv2),(1.0-u)*weight);
		//            _SmoothCons.add_row(rowCount,_x_index(coordv3),u*weight);
		//            _SmoothCons.add_row(rowCount,_y_index(coordv3),v*weight);
		//            _SmoothCons.add_row(rowCount,_y_index(coordv2),(-1.0*v)*weight);
		//            _SmoothCons.add_row(rowCount,_x_index(coordv1),(-1.0)*weight);

		//V1.x =  V2.x + u * (V3.x - V2.x) + v * (V2.y - V3.y);
		float* ptr = _SmoothConstraints.ptr<float>(_SCc);
		ptr[0] = _rowCount; ptr[1] = _x_index[coordv2]; ptr[2] = (1.0-u)*weight;  _SCc = _SCc + 1;
		ptr = _SmoothConstraints.ptr<float>(_SCc);
		ptr[0] = _rowCount; ptr[1] = _x_index[coordv3]; ptr[2] = u*weight;  _SCc = _SCc + 1;
		ptr = _SmoothConstraints.ptr<float>(_SCc);
		ptr[0] = _rowCount; ptr[1] = _y_index[coordv3]; ptr[2] = v*weight;  _SCc = _SCc + 1;
		ptr = _SmoothConstraints.ptr<float>(_SCc);
		ptr[0] = _rowCount; ptr[1] = _y_index[coordv2]; ptr[2] = -1.0*v*weight;  _SCc = _SCc + 1;
		ptr = _SmoothConstraints.ptr<float>(_SCc);
		ptr[0] = _rowCount; ptr[1] = _x_index[coordv1]; ptr[2] = -1.0*weight;  _SCc = _SCc + 1;

		_rowCount = _rowCount+1; 


		//            _SmoothCons.add_row(rowCount,_y_index(coordv2),(1.0-u)*weight);
		//            _SmoothCons.add_row(rowCount,_y_index(coordv3),u*weight);
		//            _SmoothCons.add_row(rowCount,_x_index(coordv3),v*weight);
		//            _SmoothCons.add_row(rowCount,_x_index(coordv2),(-1.0*v)*weight);
		//            _SmoothCons.add_row(rowCount,_y_index(coordv1),(-1.0)*weight);
		//            rowCount = rowCount+1;

		//V1.y = V2.y + u * (V3.y - V2.y) + v * (V3.x - V2.x);
		ptr = _SmoothConstraints.ptr<float>(_SCc);
		ptr[0] = _rowCount; ptr[1] = _y_index[coordv2]; ptr[2] = (1.0-u)*weight;  _SCc = _SCc + 1;
		ptr = _SmoothConstraints.ptr<float>(_SCc);
		ptr[0] = _rowCount; ptr[1] = _y_index[coordv3]; ptr[2] = u*weight;  _SCc = _SCc + 1;
		ptr = _SmoothConstraints.ptr<float>(_SCc);
		ptr[0] = _rowCount; ptr[1] = _x_index[coordv2]; ptr[2] = v*weight;  _SCc = _SCc + 1;
		ptr = _SmoothConstraints.ptr<float>(_SCc);
		ptr[0] = _rowCount; ptr[1] = _x_index[coordv3]; ptr[2] = -1.0*v*weight;  _SCc = _SCc + 1;
		ptr = _SmoothConstraints.ptr<float>(_SCc);
		ptr[0] = _rowCount; ptr[1] = _y_index[coordv1]; ptr[2] = -1.0*weight;  _SCc = _SCc + 1;

		_rowCount = _rowCount+1; 



	}
	void addCoefficient_7(int i, int j, float weight)
	{
		//             V1(i,j)
		//             |
		//             |
		//             _______|
		//             V3      V2
		//             (i+1,j-1)
		cv::Point2f V1 = _source->getVertex(i,j);
		cv::Point2f  V2 = _source->getVertex(i+1,j);
		cv::Point2f  V3 = _source->getVertex(i+1,j-1);

		float u,v;
		getSmoothWeight(V1,V2,V3,u,v);

		int coordv1 =  i*_width+j;
		int coordv2 =  (i+1)*_width+j;
		int coordv3 = (i+1)*_width+j-1;            


		//            _SmoothCons.add_row(rowCount,_x_index(coordv2),(1.0-u)*weight);
		//            _SmoothCons.add_row(rowCount,_x_index(coordv3),u*weight);
		//            _SmoothCons.add_row(rowCount,_y_index(coordv2),v*weight);
		//            _SmoothCons.add_row(rowCount,_y_index(coordv3),(-1.0*v)*weight);
		//            _SmoothCons.add_row(rowCount,_x_index(coordv1),(-1.0)*weight);

		//V1.x =  V2.x + u * (V3.x - V2.x) + v * (V2.y - V3.y);
		float* ptr = _SmoothConstraints.ptr<float>(_SCc);
		ptr[0] = _rowCount; ptr[1] = _x_index[coordv2]; ptr[2] = (1.0-u)*weight;  _SCc = _SCc + 1;
		ptr = _SmoothConstraints.ptr<float>(_SCc);
		ptr[0] = _rowCount; ptr[1] = _x_index[coordv3]; ptr[2] = u*weight;  _SCc = _SCc + 1;
		ptr = _SmoothConstraints.ptr<float>(_SCc);
		ptr[0] = _rowCount; ptr[1] = _y_index[coordv2]; ptr[2] = v*weight;  _SCc = _SCc + 1;
		ptr = _SmoothConstraints.ptr<float>(_SCc);
		ptr[0] = _rowCount; ptr[1] = _y_index[coordv3]; ptr[2] = -1.0*v*weight;  _SCc = _SCc + 1;
		ptr = _SmoothConstraints.ptr<float>(_SCc);
		ptr[0] = _rowCount; ptr[1] = _x_index[coordv1]; ptr[2] = -1.0*weight;  _SCc = _SCc + 1;

		_rowCount = _rowCount+1; 


		//            _SmoothCons.add_row(rowCount,_y_index(coordv2),(1.0-u)*weight);
		//            _SmoothCons.add_row(rowCount,_y_index(coordv3),u*weight);
		//            _SmoothCons.add_row(rowCount,_x_index(coordv3),v*weight);
		//            _SmoothCons.add_row(rowCount,_x_index(coordv2),(-1.0*v)*weight);
		//            _SmoothCons.add_row(rowCount,_y_index(coordv1),(-1.0)*weight);
		//            rowCount = rowCount+1;

		//V1.y = V2.y + u * (V3.y - V2.y) + v * (V3.x - V2.x);
		ptr = _SmoothConstraints.ptr<float>(_SCc);
		ptr[0] = _rowCount; ptr[1] = _y_index[coordv2]; ptr[2] = (1.0-u)*weight;  _SCc = _SCc + 1;
		ptr = _SmoothConstraints.ptr<float>(_SCc);
		ptr[0] = _rowCount; ptr[1] = _y_index[coordv3]; ptr[2] = u*weight;  _SCc = _SCc + 1;
		ptr = _SmoothConstraints.ptr<float>(_SCc);
		ptr[0] = _rowCount; ptr[1] = _x_index[coordv3]; ptr[2] = v*weight;  _SCc = _SCc + 1;
		ptr = _SmoothConstraints.ptr<float>(_SCc);
		ptr[0] = _rowCount; ptr[1] = _x_index[coordv2]; ptr[2] = -1.0*v*weight;  _SCc = _SCc + 1;
		ptr = _SmoothConstraints.ptr<float>(_SCc);
		ptr[0] = _rowCount; ptr[1] = _y_index[coordv1]; ptr[2] = -1.0*weight;  _SCc = _SCc + 1;

		_rowCount = _rowCount+1; 



	}
	void addCoefficient_8(int i, int j, float weight)
	{
		//             V2        V1(i,j)
		//             _________
		//             |
		//             |
		//             |
		//             V3(i+1,j-1)
		cv::Point2f V1 = _source->getVertex(i,j);
		cv::Point2f  V2 = _source->getVertex(i,j-1);
		cv::Point2f  V3 = _source->getVertex(i+1,j-1);

		float u,v;
		getSmoothWeight(V1,V2,V3,u,v);

		int coordv1 =  i*_width+j;
		int coordv2 =  i*_width+j-1;
		int coordv3 = (i+1)*_width+j-1;            


		//            _SmoothCons.add_row(rowCount,_x_index(coordv2),(1.0-u)*weight);
		//            _SmoothCons.add_row(rowCount,_x_index(coordv3),u*weight);
		//            _SmoothCons.add_row(rowCount,_y_index(coordv2),v*weight);
		//            _SmoothCons.add_row(rowCount,_y_index(coordv3),(-1.0*v)*weight);
		//            _SmoothCons.add_row(rowCount,_x_index(coordv1),(-1.0)*weight);

		//V1.x =  V2.x + u * (V3.x - V2.x) + v * (V2.y - V3.y);
		float* ptr = _SmoothConstraints.ptr<float>(_SCc);
		ptr[0] = _rowCount; ptr[1] = _x_index[coordv2]; ptr[2] = (1.0-u)*weight;  _SCc = _SCc + 1;
		ptr = _SmoothConstraints.ptr<float>(_SCc);
		ptr[0] = _rowCount; ptr[1] = _x_index[coordv3]; ptr[2] = u*weight;  _SCc = _SCc + 1;
		ptr = _SmoothConstraints.ptr<float>(_SCc);
		ptr[0] = _rowCount; ptr[1] = _y_index[coordv3]; ptr[2] = v*weight;  _SCc = _SCc + 1;
		ptr = _SmoothConstraints.ptr<float>(_SCc);
		ptr[0] = _rowCount; ptr[1] = _y_index[coordv2]; ptr[2] = -1.0*v*weight;  _SCc = _SCc + 1;
		ptr = _SmoothConstraints.ptr<float>(_SCc);
		ptr[0] = _rowCount; ptr[1] = _x_index[coordv1]; ptr[2] = -1.0*weight;  _SCc = _SCc + 1;

		_rowCount = _rowCount+1; 


		//            _SmoothCons.add_row(rowCount,_y_index(coordv2),(1.0-u)*weight);
		//            _SmoothCons.add_row(rowCount,_y_index(coordv3),u*weight);
		//            _SmoothCons.add_row(rowCount,_x_index(coordv3),v*weight);
		//            _SmoothCons.add_row(rowCount,_x_index(coordv2),(-1.0*v)*weight);
		//            _SmoothCons.add_row(rowCount,_y_index(coordv1),(-1.0)*weight);
		//            rowCount = rowCount+1;

		//V1.y = V2.y + u * (V3.y - V2.y) + v * (V3.x - V2.x);
		ptr = _SmoothConstraints.ptr<float>(_SCc);
		ptr[0] = _rowCount; ptr[1] = _y_index[coordv2]; ptr[2] = (1.0-u)*weight;  _SCc = _SCc + 1;
		ptr = _SmoothConstraints.ptr<float>(_SCc);
		ptr[0] = _rowCount; ptr[1] = _y_index[coordv3]; ptr[2] = u*weight;  _SCc = _SCc + 1;
		ptr = _SmoothConstraints.ptr<float>(_SCc);
		ptr[0] = _rowCount; ptr[1] = _x_index[coordv2]; ptr[2] = v*weight;  _SCc = _SCc + 1;
		ptr = _SmoothConstraints.ptr<float>(_SCc);
		ptr[0] = _rowCount; ptr[1] = _x_index[coordv3]; ptr[2] = -1.0*v*weight;  _SCc = _SCc + 1;
		ptr = _SmoothConstraints.ptr<float>(_SCc);
		ptr[0] = _rowCount; ptr[1] = _y_index[coordv1]; ptr[2] = -1.0*weight;  _SCc = _SCc + 1;

		_rowCount = _rowCount+1; 



	}
	void CreateSmoothCons(float weight);
	void Solve();
private:
	int  _height;
	int _width;
	int _quadWidth;
	int _quadHeight;
	float _weight;
	std::vector<float> _V00, _V01, _V10, _V11;
	std::vector<cv::Point2f> _orgPts, _destPts;
	std::vector<int> _x_index, _y_index;
	Mesh* _source, *_destin;
	int _columns;
	
	//smooth constraints
	int _num_smooth_cons;	
	cv::Mat _SmoothConstraints;
	int _SCc;

	//data constraints
	std::vector<float>    _dataterm_element_i;
	std::vector<float>    _dataterm_element_j;
	std::vector<cv::Point2f>   _dataterm_element_orgPt;
	std::vector<cv::Point2f>    _dataterm_element_desPt;
	std::vector<float>   _dataterm_element_V00;
	std::vector<float>   _dataterm_element_V01;
	std::vector<float>  _dataterm_element_V10;
	std::vector<float>   _dataterm_element_V11;
	cv::Mat _DataConstraints;
	int _DCc;
	int _num_data_cons;

	int _rowCount;
};