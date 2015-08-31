#pragma once
#include <vector>
#include "CudaSuperpixel.h"
#include "Common.h"
using namespace std;

class ComSuperpixel
{
public:
	ComSuperpixel()
	{
		m_rvec = m_gvec = m_bvec = NULL;
	};
	~ComSuperpixel()
	{
		safe_delete_array(m_rvec);
		safe_delete_array(m_gvec);
		safe_delete_array(m_bvec);
	}
	void DetectRGBEdges(
		const double*				rvec,
		const double*				gvec,
		const double*				bvec,
		const int&					width,
		const int&					height,
		vector<double>&				edges);
	void GetRGBXYSeeds_ForGivenK(
	vector<double>&				kseedsl,
	vector<double>&				kseedsa,
	vector<double>&				kseedsb,
	vector<double>&				kseedsx,
	vector<double>&				kseedsy,
	const int&					K,
	const bool&					perturbseeds,
	const vector<double>&		edgemag);
void GetRGBXYSeeds_ForGivenStep(
	vector<double>&				kseedsl,
	vector<double>&				kseedsa,
	vector<double>&				kseedsb,
	vector<double>&				kseedsx,
	vector<double>&				kseedsy,
	const int&					S,
	const bool&					perturbseeds,
	const vector<double>&		edgemag);
	void PerturbSeeds(
	vector<double>&				kseedsl,
	vector<double>&				kseedsa,
	vector<double>&				kseedsb,
	vector<double>&				kseedsx,
	vector<double>&				kseedsy,
	const vector<double>&		edges);
	double Distance(unsigned x,unsigned y, int labelIdx,vector<double>&				kseedsl,
	vector<double>&				kseedsa,
	vector<double>&				kseedsb,
	vector<double>&				kseedsx,
	vector<double>&				kseedsy)
	{
		
		int idx = x + y*m_width;
		double dr = abs(m_rvec[idx] - kseedsl[labelIdx]);
		double dg = abs(m_gvec[idx] - kseedsa[labelIdx]) ;
		double db = abs(m_bvec[idx] - kseedsb[labelIdx]);
		double d_rgb = sqrt(dr*dr + dg*dg + db*db);
		double dx = abs(x - kseedsx[labelIdx]);
		double dy =  abs(y - kseedsy[labelIdx]);
		double d_xy = sqrt(dx*dx + dy*dy);
		return (1-m_alpha)*d_rgb/255 + m_alpha*d_xy/(2*m_radius);
	}
	void Superpixel(unsigned * rgbBuffer,unsigned width, unsigned height, int num, float alpha,int* lables);
	void Superpixel(unsigned * rgbBuffer,unsigned width, unsigned height, int step, float alpha,int& num,int* lables);
	void SuperpixelLattice(unsigned * rgbBuffer,unsigned width, unsigned height, int step, float alpha,int& num,int* lables);
	void SuperpixelLattice(uchar4 * rgbBuffer,unsigned width, unsigned height, int step, float alpha,int& num,int* lables,SLICClusterCenter* centers);
private:
	unsigned m_height;
	unsigned m_width;
	double* m_rvec;
	double* m_gvec;
	double* m_bvec;
	float m_alpha;
	int* m_labels;
	int m_nSuperpixels;
	double m_radius;
};