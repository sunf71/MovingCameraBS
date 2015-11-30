#pragma once
#include <vector>
#include <stdlib.h>
#include "../QC/sparse_matlab_like_matrix.hpp"
#include "../QC/QC_full_sparse.hpp"
#include "../QC/ind_sim_pair.hpp"
#include "../QC/sparse_similarity_matrix_utils.hpp"
#include <opencv\cv.h>

class HistComparer
{
public:
	virtual double Distance(const std::vector<float>& h1, const std::vector<float>& h2) = 0;
};
class QCHistComparer :public HistComparer
{
public:
	QCHistComparer(int N, float normF = 0.9, int threshold = 3) :_normF(normF), _N(N),_threshold(threshold)
	{
		// Similarity matrix
		_A.resize(N);
		for (int i = 0; i<N; ++i) _A[i].push_back(ind_sim_pair(i, 1.0));
		//sparse_similarity_matrix_utils::insert_into_A_symmetric_sim(A, 0, 1, 0.2); // A(0,1)= 0.2 and A(1,0)= 0.2
		//sparse_similarity_matrix_utils::insert_into_A_symmetric_sim(A, 0, 2, 0.1);
		//sparse_similarity_matrix_utils::insert_into_A_symmetric_sim(A, 3, 4, 0.2);
		for (size_t i = 0; i < N; i++)
		{
			
			int max = std::min(N - 1, (int)(i + threshold-1));
			for (size_t j = i+1; j <= max; j++)
			{
				sparse_similarity_matrix_utils::insert_into_A_symmetric_sim(_A, i, j, 1 - (abs((int)(i - j)) *1.0 / threshold));
			}
		}
	}
	
	virtual double Distance(const std::vector<float>& h1, const std::vector<float>& h2)
	{
		assert(h1.size() == h2.size() && h1.size() == _N);
		return _qc_full_sparse(&h1[0], &h2[0], _A, _normF, _N);
	}
private:
	int _N;
	float _normF;
	int _threshold;
	// Similarity matrix
	std::vector< std::vector<ind_sim_pair> > _A;
	QC_full_sparse _qc_full_sparse;
};

class CVBHATTHistComparer :public HistComparer
{
public:
	virtual  double Distance(const std::vector<float>& h1, const std::vector<float>& h2)
	{
		return cv::compareHist(h1, h2, CV_COMP_BHATTACHARYYA);
	}

};