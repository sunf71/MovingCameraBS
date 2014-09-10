#include "ComSuperpixel.h"
#include <math.h>
#include <iostream>
void ComSuperpixel::Superixel(unsigned int * rgbBuffer,unsigned width, unsigned height, int num, float alpha,int* labels)
{
	//initialize
	vector<double> kseedsr(0);
	vector<double> kseedsg(0);
	vector<double> kseedsb(0);
	vector<double> kseedsx(0);
	vector<double> kseedsy(0);

	//--------------------------------------------------
	m_width  = width;
	m_height = height;
	m_alpha = alpha;
	m_labels = labels;
	int sz = m_width*m_height;
	//--------------------------------------------------
	//if(0 == klabels) klabels = new int[sz];
	for( int s = 0; s < sz; s++ ) labels[s] = -1;
	//--------------------------------------------------

	m_rvec = new double[sz];
	m_gvec = new double[sz];
	m_bvec = new double[sz];

	for( int j = 0; j < sz; j++ )
	{
		m_rvec[j] = (rgbBuffer[j] >> 16) & 0xFF;
		m_gvec[j] = (rgbBuffer[j] >>  8) & 0xFF;
		m_bvec[j] = (rgbBuffer[j]      ) & 0xFF;


	}

	vector<double> edgemag(0);
	DetectRGBEdges(m_rvec, m_gvec, m_bvec, m_width, m_height, edgemag);
	GetRGBXYSeeds_ForGivenK(kseedsr,kseedsg,kseedsb,kseedsx,kseedsy,num,true,edgemag);

	//iteration
	int itr = 0;
	const int dx8[8] = {-1, -1,  0,  1, 1, 1, 0, -1};
	const int dy8[8] = { 0, -1, -1, -1, 0, 1, 1,  1};
	while(itr < 1)
	{
		itr++;
		
		vector<bool> istaken(sz, false);

		int mainindex(0);
		for( int j = 0; j < m_height; j++ )
		{
			for( int k = 0; k < m_width; k++ )
			{
				int np(0);
				std::vector<int> nl;
				for( int i = 0; i < 8; i++ )
				{
					int x = k + dx8[i];
					int y = j + dy8[i];

					if( (x >= 0 && x < m_width) && (y >= 0 && y < m_height) )
					{
						int index = y*m_width + x;

						if( false == istaken[index] )//comment this to obtain internal contours
						{
							if( labels[mainindex] != labels[index] ) 
							{
								np++;
								nl.push_back(labels[index]);
							}
						}
					}
				}
				if( np > 1 )//change to 2 or 3 for thinner lines
				{
					double min = Distance(k,j,labels[mainindex],kseedsr,kseedsb,kseedsb,kseedsx,kseedsy);
					int idx = 0;
					for(int i=0; i<nl.size(); i++)
					{
						double dis = Distance(k,j,nl[i],kseedsr,kseedsb,kseedsb,kseedsx,kseedsy);
						if (dis < min)
						{
							min = dis;
							idx = i;
						}
					}
					labels[mainindex] = nl[idx];
				}
				mainindex++;
				
			}
		}
	}
}

void ComSuperpixel::PerturbSeeds(
	vector<double>&				kseedsl,
	vector<double>&				kseedsa,
	vector<double>&				kseedsb,
	vector<double>&				kseedsx,
	vector<double>&				kseedsy,
	const vector<double>&		edges)
{
	const int dx8[8] = {-1, -1,  0,  1, 1, 1, 0, -1};
	const int dy8[8] = { 0, -1, -1, -1, 0, 1, 1,  1};

	int numseeds = kseedsl.size();

	for( int n = 0; n < numseeds; n++ )
	{
		int ox = kseedsx[n];//original x
		int oy = kseedsy[n];//original y
		int oind = oy*m_width + ox;

		int storeind = oind;
		for( int i = 0; i < 8; i++ )
		{
			int nx = ox+dx8[i];//new x
			int ny = oy+dy8[i];//new y

			if( nx >= 0 && nx < m_width && ny >= 0 && ny < m_height)
			{
				int nind = ny*m_width + nx;
				if( edges[nind] < edges[storeind])
				{
					storeind = nind;
				}
			}
		}
		if(storeind != oind)
		{
			kseedsx[n] = storeind%m_width;
			kseedsy[n] = storeind/m_width;
			kseedsl[n] = m_rvec[storeind];
			kseedsa[n] = m_gvec[storeind];
			kseedsb[n] = m_bvec[storeind];
		}
	}
}

void ComSuperpixel::GetRGBXYSeeds_ForGivenK(
	vector<double>&				kseedsl,
	vector<double>&				kseedsa,
	vector<double>&				kseedsb,
	vector<double>&				kseedsx,
	vector<double>&				kseedsy,
	const int&					K,
	const bool&					perturbseeds,
	const vector<double>&		edgemag)
{
	int sz = m_width*m_height;
	double step = sqrt(double(sz)/double(K));
	int T = step;
	int xoff = step/2;
	int yoff = step/2;

	int n(0);int r(0);
	for( int y = 0; y < m_height; y++ )
	{
		int Y = y*step + yoff;
		if( Y > m_height-1 ) break;

		for( int x = 0; x < m_width; x++ )
		{
			int X = x*step + xoff;//square grid
			//int X = x*step + (xoff<<(r&0x1));//hex grid
			if(X > m_width-1) break;

			int i = Y*m_width + X;

			//_ASSERT(n < K);

			//kseedsl[n] = m_lvec[i];
			//kseedsa[n] = m_avec[i];
			//kseedsb[n] = m_bvec[i];
			//kseedsx[n] = X;
			//kseedsy[n] = Y;
			kseedsl.push_back(m_rvec[i]);
			kseedsa.push_back(m_gvec[i]);
			kseedsb.push_back(m_bvec[i]);
			kseedsx.push_back(X);
			kseedsy.push_back(Y);

			for(int k= X - xoff; k<= X + xoff; k++)
			{
				if (k>m_width-1)
					break;
				for(int j=Y-yoff; j<= Y+yoff; j++)
				{
					if (j>m_height-1)
						break;
					m_labels[k + j*m_width] = n;
				}
			}
			n++;
		}
		r++;
	}

	if(perturbseeds)
	{
		PerturbSeeds(kseedsl, kseedsa, kseedsb, kseedsx, kseedsy, edgemag);
	}
	
}

void ComSuperpixel::DetectRGBEdges(const double*				lvec,
	const double*				avec,
	const double*				bvec,
	const int&					width,
	const int&					height,
	vector<double>&				edges)
{
	int sz = width*height;

	edges.resize(sz,0);
	for( int j = 1; j < height-1; j++ )
	{
		for( int k = 1; k < width-1; k++ )
		{
			int i = j*width+k;

			double dx = (lvec[i-1]-lvec[i+1])*(lvec[i-1]-lvec[i+1]) +
				(avec[i-1]-avec[i+1])*(avec[i-1]-avec[i+1]) +
				(bvec[i-1]-bvec[i+1])*(bvec[i-1]-bvec[i+1]);

			double dy = (lvec[i-width]-lvec[i+width])*(lvec[i-width]-lvec[i+width]) +
				(avec[i-width]-avec[i+width])*(avec[i-width]-avec[i+width]) +
				(bvec[i-width]-bvec[i+width])*(bvec[i-width]-bvec[i+width]);

			//edges[i] = (sqrt(dx) + sqrt(dy));
			edges[i] = (dx + dy);
		}
	}
}