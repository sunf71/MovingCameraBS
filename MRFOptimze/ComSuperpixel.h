#pragma once
#include <vector>
#define safe_delete(X) if (X!=NULL) delete[] X;
using namespace std;

class ComSuperpixel
{
public:
	ComSuperpixel(){};
	~ComSuperpixel()
	{
		safe_delete(m_rvec);
		safe_delete(m_gvec);
		safe_delete(m_bvec);
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
		return (1-m_alpha)*d_rgb + m_alpha*d_xy/m_radius;
	}
	void Superixel(unsigned * rgbBuffer,unsigned width, unsigned height, int num, float alpha,int* lables);
private:
	unsigned m_height;
	unsigned m_width;
	double* m_rvec;
	double* m_gvec;
	double* m_bvec;
	float m_alpha;
	int* m_labels;
	double m_radius;
};