#pragma once
#include <vector>
#include <algorithm>
using namespace std;

//统计类，用于计算标准差，协方差等统计数据
class STAT
{
public:

	STAT(void)
	{
	}

	~STAT(void)
	{
	}

	//计算均值 
	template  <class T>
	static double Mean(vector<T>& v)
	{
		double sum = 0;
		for(int i=0; i<v.size(); i++)
		{
			sum += v[i];
		}
		return sum/v.size();
	}
	//计算方差
	template <class T>
	static double Variance(vector<T>& v)
	{
		double ret = 0;
		double mean = Mean(v);
		for(int i=0; i<v.size(); i++)
		{
			ret += (v[i] - mean)*(v[i] - mean);
		}
		return ret/v.size();
	}
	template <class T>
	static double Variance(vector<T>& v,double mean)
	{
		double ret = 0;		
		for(int i=0; i<v.size(); i++)
		{
			ret += (v[i] - mean)*(v[i] - mean);
		}
		return ret/v.size();
	}
	//计算标准差
	template <class T>
	static double STD(vector<T>& v)
	{
		return sqrt(Variance(v));
	}

	//计算协方差
	template <class T>
	static double Cov(vector<T>& a, vector<T>& b)
	{
		assert(a.size() == b.size());

		//cov(a,b) = E(a*b) - E(a)*E(b)
		double ret = 0;
		double suma = 0;
		double sumb = 0;
		for(int i=0; i<a.size(); i++)
		{
			ret += a[i]*b[i];
			suma += a[i];
			sumb += b[i];		
		}
		
		return (ret - suma*sumb/a.size())/a.size();
	}

	template <class T>
	static double Cov(vector<T>& a, vector<T>&b, double meanA, double meanB)
	{
		assert(a.size() == b.size());
		double ret = 0;
		
		for(int i=0; i<a.size(); i++)
		{
			ret += a[i]*b[i];
			
		}
		return ret/a.size() - meanA*meanB;
	}

	template<class T>
	static double BhatBinDistance(const vector<T>& a, const vector<T>& b)
	{
		assert(a.size() == b.size());
		float sumXY = 0;
		float sumX = 0;
		float sumY = 0;
		for(int i=0; i<a.size(); i++)
		{
			sumXY += sqrt(a[i]*b[i]);
			sumX += a[i];
			sumY += b[i];
		}

		return 1 - sumXY/sqrt(sumX*sumY);
	}

	//计算直方图距离
	template <class T>
	static double BinDistance(const vector<T>& a, const vector<T>& b)
	{
		return 1- abs(CorDistance(a,b));
	/*	return BhatBinDistance(a,b);*/
	}

	//计算直方图距离
	template <class T>
	static double BinDistance(const vector<vector<T>>& a, const vector<vector<T>>& b)
	{
		
		double Max = 0;		
		int c = 0;
	    vector<T> sumA(a[0].size(),0),sumB(a[0].size(),0);

		for(int i=0; i<3; i++)
		{
		/*	double max = *max_element(a[i].begin(),a[i].end());
			double ratio = max/accumulate(a[i].begin(),a[i].end(),0);
			max *= ratio;
			if (max > Max)
			{
				Max = max;
				c = i;
			}*/
			for(int j=0; j<a[0].size(); j++)
			{
				sumA[j] += a[i][j];
				sumB[j] += b[i][j];
			}
		}	
		return BinDistance(sumA,sumB);
		//return BinDistance(a[c],b[c]);		
	}
	//计算直方图距离
	template <class T>
	static double CorDistance(const vector<T>& a, const vector<T>& b)
	{	
		//distance = (cov(a,b) + c)/(std(a)*std(b) + c) c是一个防止除0的常数 0.00001
		const double c = 0.00001;
		assert(a.size() == b.size());

		//cov(a,b) = E(a*b) - E(a)*E(b)
		double cov = 0;
		double suma = 0;
		double sumb = 0;
		for(int i=0; i<a.size(); i++)
		{
			cov += a[i]*b[i];
			suma += a[i];
			sumb += b[i];		
		}
		
		double meanA = suma/a.size();
		double meanB = sumb/b.size();

		double stdA = 0;
		double stdB = 0;
		for(int i=0; i<a.size(); i++)
		{
			stdA += (a[i] - meanA)*(a[i] - meanA);
			stdB += (b[i] - meanB)*(b[i] - meanB);

		}
		stdA = sqrt(stdA/a.size());
		stdB = sqrt(stdB/a.size());

		cov /= a.size();
		cov -= meanA * meanB;
		
		return (cov + c)/(stdA*stdB +c);	
	
	}
};