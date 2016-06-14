#include "SparseSolver.h"
#include <fstream>
void checkStatus(culaStatus status)
{
    char buf[256];

    if(!status)
        return;

    culaGetErrorInfoString(status, culaGetErrorInfo(), buf, sizeof(buf));
    printf("%s\n", buf);

    culaShutdown();
    exit(EXIT_FAILURE);
}
//cmat 是3行n列矩阵，每一行中前两列表示数据所处的行和列，最后一列表示数据的值
void SolveSparse(const cv::Mat& cmat, std::vector<double>& rhs, std::vector<double>& result)
{
	std::vector<int>jn,ia;
	std::vector<double> an;
	std::vector<float> r,c;
	std::vector<float> v;
	r.resize(cmat.rows);
	c.resize(cmat.rows);
	v.resize(cmat.rows);
	cv::Mat tcmat;
	cv::transpose(cmat,tcmat);
	tcmat.row(0).copyTo(r);
	tcmat.row(1).copyTo(c);
	tcmat.row(2).copyTo(v);
	int k=0;
	/*std::ofstream sfile("amat.txt");*/
	for(int i=0; i<v.size(); i++)
	{
		if (abs(v[i]) > 1e-6)
		{
			r[k] = r[i];
			c[k] = c[i];
			v[k] = v[i];
			
			k++;
		}
	}
	
	r.resize(k);
	c.resize(k);
	v.resize(k);
	std::vector<int> idx(r.size());
	for(int i=0; i<idx.size(); i++)
		idx[i] = i;
	MyAcsdRCComp<float> myARCom(&c,&r);
	std::sort(idx.begin(),idx.end(),myARCom);
	int pos(0);
	int lastC(-1);
	int maxR(0);
	for(int i=0; i<idx.size(); i++)
	{
		//sfile<<"("<<r[idx[i]]<<","<<c[idx[i]]<<")\t"<<v[idx[i]]<<std::endl;
		if (abs(c[idx[i]] - lastC) > 1e-6)
		{
			lastC = c[idx[i]];			
			ia.push_back(pos);			
		}
		if (r[idx[i]] > maxR)
			maxR = r[idx[i]];
		jn.push_back(r[idx[i]]);
		an.push_back(v[idx[i]]);
		pos++;
	}
	ia.push_back(pos);
	/*std::ofstream file("sparsemat.txt");
	file<<"ia\n";
	for(int i=0; i<ia.size(); i++)
	{
		file<<ia[i]<<std::endl;
	}
	file<<"\njn\n;";
	for(int i=0; i<jn.size(); i++)
	{
		file<<jn[i]<<std::endl;
	}
	file<<"\nan\n";
	for(int i=0; i<an.size(); i++)
	{
		file<<an[i]<<std::endl;
	}
	file.close();
	sfile.close();
	*/
	int    m = rhs.size();
	int	n = ia.size()-1;
	cv::Mat fullA = cv::Mat::zeros(m,n,CV_32F);
	for(int i=0; i<idx.size(); i++)
	{
		fullA.at<float>(r[idx[i]],c[idx[i]]) = v[idx[i]];
	}
	cv::Mat bMat(rhs.size(),1,CV_32F);
	cv::Mat xMat(n,1,CV_32F);
	for(int i=0; i<rhs.size(); i++)
		bMat.at<float>(i,0) = rhs[i];
	cv::solve(fullA,bMat,xMat,cv::DECOMP_SVD);
	result.resize(n);
	/*std::ofstream xfile("x.txt");*/
	for(int i=0; i<n; i++)
	{
		result[i] = xMat.at<float>(i,0);
		//xfile<<result[i]<<std::endl;
	}
	//xfile.close();
	/*int    *Ap = &ia[0];
	int    *Ai =&jn[0];
	double* Ax = &an[0];
	double *b = &rhs[0];
	
	double *x = &result[0];
	double *null = (double *) NULL ;
   
    void *Symbolic, *Numeric ;
    (void) umfpack_di_symbolic (m, n, Ap, Ai, Ax, &Symbolic, null, null) ;
    (void) umfpack_di_numeric (Ap, Ai, Ax, Symbolic, &Numeric, null, null) ;
    umfpack_di_free_symbolic (&Symbolic) ;
    (void) umfpack_di_solve (UMFPACK_A, Ap, Ai, Ax, x, b, Numeric, null, null) ;
    umfpack_di_free_numeric (&Numeric) ;*/

}

//Amat 是x行3列矩阵，每一行中前两列表示数据所处的行和列，最后一列表示数据的值
//Bmat 是m行1列矩阵 解线性方程组Ax = b其中 A是m行n列矩阵（m>=n), b是bMat中的m行1列向量
void LeastSquareSolve(const int m, const int n, const cv::Mat& Amat, const cv::Mat& bMat, std::vector<float>& result)
{
	culaStatus status;
	status = culaInitialize();
	checkStatus(status);
	float*  A = new float[m*n];
	memset(A,0,sizeof(float)*m*n);
	//std::vector<float> A(m*n);
	for(int i=0; i< Amat.rows; i++)
	{
		const float* ptr = Amat.ptr<float>(i);
		int row = ptr[0];
		int col = ptr[1];
		A[col*m+row] = ptr[2];
	}
	float* b = new float[m];	
	memcpy(b,bMat.data,sizeof(float)*m);

	status = culaSgels('N', m, n, 1, &A[0], m, &b[0], m);
	checkStatus(status);
	result.resize(n);
	memcpy(&result[0],b,sizeof(float)*n);
	delete[] A;
	delete[] b;
}


void CvLeastSquareSolve(const int m, const int n, const cv::Mat& Amat, const cv::Mat& bMat, std::vector<float>& result)
{
	
	cv::Mat fullA = cv::Mat::zeros(m,n,CV_32F);
	for(int i=0; i< Amat.rows; i++)
	{
		const float* ptr = Amat.ptr<float>(i);
		int row = ptr[0];
		int col = ptr[1];
		fullA.at<float>(row,col) = ptr[2];
	}
	
	
	cv::Mat xMat(n,1,CV_32F);
	
	cv::solve(fullA,bMat,xMat,cv::DECOMP_SVD);
	result.resize(n);
	xMat.col(0).copyTo(result);
}