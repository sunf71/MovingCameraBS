#include "ASAPWarping.h"
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
		cv::Point2f pt = inputsPts[i];
		_dataterm_element_i[i] = floor(pt.y/_quadHeight)+1;
		_dataterm_element_j[i] = floor(pt.x/_quadWidth)+1;

		Quad qd = _source->getQuad(_dataterm_element_i[i],_dataterm_element_j[i]);


		
		qd.getBilinearCoordinates(pt,coefficients);
		_dataterm_element_V00[i] = coefficients[1];
		_dataterm_element_V01[i] = coefficients[2];
		_dataterm_element_V10[i] = coefficients[3];
		_dataterm_element_V11[i] = coefficients[4];
	}

}

void ASAPWarping::CreateDataCons(cv::Mat& b)
{
	int len = _dataterm_element_i.size();
	_num_data_cons = len*2;
	_DataConstraints.create(_num_data_cons,3,CV_32F);
	b.create(_num_data_cons+_num_smooth_cons,1,CV_32F);
	_DCc = 0;
	for(int k=0; k<len; k++)
	{
		float i = _dataterm_element_i[k];
		float j = _dataterm_element_j[k];
		float v00 = _dataterm_element_V00[k];
		float v01 = _dataterm_element_V01[k];
		float v10 = _dataterm_element_V10[k];
		float v11 = _dataterm_element_V11[k];
		float * ptr = _DataConstraints.ptr<float>(_DCc);       
		ptr[0] = _rowCount; ptr[1] = _x_index[(i-1)*_width+j-1+1]; ptr[2]= v00; _DCc = _DCc+1;
		ptr = _DataConstraints.ptr<float>(_DCc); 
		ptr[0] = _rowCount; ptr[1] = _x_index[(i-1)*_width+j+1]; ptr[2]= v01; _DCc = _DCc+1;
		ptr = _DataConstraints.ptr<float>(_DCc); 
		ptr[0] = _rowCount; ptr[1] = _x_index[i*_width+j-1+1]; ptr[2]= v10; _DCc = _DCc+1;
		ptr = _DataConstraints.ptr<float>(_DCc); 
		ptr[0] = _rowCount; ptr[1] = _x_index[i*_width+j+1]; ptr[2]= v11; _DCc = _DCc+1;		
		_rowCount = _rowCount+1;
		b.at<float>(_rowCount,0) = _dataterm_element_desPt[k].x;

		ptr = _DataConstraints.ptr<float>(_DCc);       
		ptr[0] = _rowCount; ptr[1] = _y_index[(i-1)*_width+j-1+1]; ptr[2]= v00; _DCc = _DCc+1;
		ptr = _DataConstraints.ptr<float>(_DCc); 
		ptr[0] = _rowCount; ptr[1] = _y_index[(i-1)*_width+j+1]; ptr[2]= v01; _DCc = _DCc+1;
		ptr = _DataConstraints.ptr<float>(_DCc); 
		ptr[0] = _rowCount; ptr[1] = _y_index[i*_width+j-1+1]; ptr[2]= v10; _DCc = _DCc+1;
		ptr = _DataConstraints.ptr<float>(_DCc); 
		ptr[0] = _rowCount; ptr[1] = _y_index[i*_width+j+1]; ptr[2]= v11; _DCc = _DCc+1;		
		_rowCount = _rowCount+1;        
		b.at<float>(_rowCount,0) = _dataterm_element_desPt[k].x;


	}
}

void ASAPWarping::Solve()
{
	cv::Mat b;
	CreateDataCons(b);
	int N = _SmoothConstraints.rows + _DataConstraints.rows;

	cv::Mat ARows = cv::Mat::zeros(N,1,CV_32F);
	cv::Mat ACols = cv::Mat::zeros(N,1,CV_32F);
	cv::Mat AVals = cv::Mat::zeros(N,1,CV_32F);

	int cc = 0;
	for( int i=0; i<_Smooth

}
