#include "ComSuperpixel.h"
#include <math.h>
#include <iostream>
#include <opencv\cv.h>
void ComSuperpixel::Superpixel(unsigned int * rgbBuffer,unsigned width, unsigned height, int num, float alpha,int* labels)
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
	vector<double> sigmal(m_nSuperpixels, 0);
	vector<double> sigmaa(m_nSuperpixels, 0);
	vector<double> sigmab(m_nSuperpixels, 0);
	vector<double> sigmax(m_nSuperpixels, 0);
	vector<double> sigmay(m_nSuperpixels, 0);
	vector<int> clustersize(m_nSuperpixels, 0);
	vector<double> inv(m_nSuperpixels, 0);//to store 1/clustersize[k] values
	int itr = 0;
	const int dx4[4] = {-1,  0,  1, 0,};
	const int dy4[4] = { 0, -1, 0, 1};
	while(itr < 10)
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
				for( int i = 0; i < 4; i++ )
				{
					int x = k + dx4[i];
					int y = j + dy4[i];

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
				if( np > 0 )//change to 2 or 3 for thinner lines
				{
					double min = Distance(k,j,labels[mainindex],kseedsr,kseedsb,kseedsb,kseedsx,kseedsy);
					int idx = -1;
					for(int i=0; i<nl.size(); i++)
					{
						double dis = Distance(k,j,nl[i],kseedsr,kseedsb,kseedsb,kseedsx,kseedsy);
						if (dis < min)
						{
							min = dis;
							idx = i;
						}
					}
					if (idx >=0)
					labels[mainindex] = nl[idx];
				}
				mainindex++;
				//std::cout<<mainindex<<std::endl;
			}
		}

		//-----------------------------------------------------------------
		// Recalculate the centroid and store in the seed values
		//-----------------------------------------------------------------
		sigmal.assign(m_nSuperpixels, 0);
		sigmaa.assign(m_nSuperpixels, 0);
		sigmab.assign(m_nSuperpixels, 0);
		sigmax.assign(m_nSuperpixels, 0);
		sigmay.assign(m_nSuperpixels, 0);
		clustersize.assign(m_nSuperpixels, 0);

		for( int j = 0; j < sz; j++ )
		{
			int temp = labels[j];
			//std::cout<<j<<":"<<labels[j]<<std::endl;
			sigmal[labels[j]] += m_rvec[j];
			sigmaa[labels[j]] += m_gvec[j];
			sigmab[labels[j]] += m_bvec[j];
			sigmax[labels[j]] += (j%m_width);
			sigmay[labels[j]] += (j/m_width);

			clustersize[labels[j]]++;
		}

		{for( int k = 0; k < num; k++ )
		{
			//_ASSERT(clustersize[k] > 0);
			if( clustersize[k] <= 0 ) clustersize[k] = 1;
			inv[k] = 1.0/double(clustersize[k]);//computing inverse now to multiply, than divide later
		}}
		
		{for( int k = 0; k < num; k++ )
		{
			kseedsr[k] = sigmal[k]*inv[k];
			kseedsg[k] = sigmaa[k]*inv[k];
			kseedsb[k] = sigmab[k]*inv[k];
			kseedsx[k] = sigmax[k]*inv[k];
			kseedsy[k] = sigmay[k]*inv[k];
		}}
	}
}

void ComSuperpixel::Superpixel(unsigned int * rgbBuffer,unsigned width, unsigned height, int step, float alpha,int& num, int* labels)
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
	GetRGBXYSeeds_ForGivenStep(kseedsr,kseedsg,kseedsb,kseedsx,kseedsy,step,true,edgemag);
	num = m_nSuperpixels;
	//iteration
	vector<double> sigmal(m_nSuperpixels, 0);
	vector<double> sigmaa(m_nSuperpixels, 0);
	vector<double> sigmab(m_nSuperpixels, 0);
	vector<double> sigmax(m_nSuperpixels, 0);
	vector<double> sigmay(m_nSuperpixels, 0);
	vector<int> clustersize(m_nSuperpixels, 0);
	vector<double> inv(m_nSuperpixels, 0);//to store 1/clustersize[k] values
	int itr = 0;
	const int dx4[4] = {-1,  0,  1, 0,};
	const int dy4[4] = { 0, -1, 0, 1};
	while(itr < 10)
	{
		itr++;
		
		vector<bool> istaken(sz, false);

		
		for( int j = 0; j < m_height; j++ )
		{
			for( int k = 0; k < m_width; k++ )
			{
				int mainindex = k + j*m_width;
				int np(0);
				std::vector<int> nl;
				for( int i = 0; i < 4; i++ )
				{
					int x = k + dx4[i];
					int y = j + dy4[i];

					if( (x >= 0 && x < m_width) && (y >= 0 && y < m_height) )
					{
						int index = y*m_width + x;

						
						if( labels[mainindex] != labels[index] ) 
						{
							np++;
							nl.push_back(labels[index]);
						}
						

					}
				}
				if( np > 0 )//change to 2 or 3 for thinner lines
				{
					double min = Distance(k,j,labels[mainindex],kseedsr,kseedsb,kseedsb,kseedsx,kseedsy);
					int idx = -1;
					for(int i=0; i<nl.size(); i++)
					{
						double dis = Distance(k,j,nl[i],kseedsr,kseedsb,kseedsb,kseedsx,kseedsy);
						if (dis < min)
						{
							min = dis;
							idx = i;
						}
					}
					if (idx >=0)
					labels[mainindex] = nl[idx];
				}
				//mainindex++;
				//std::cout<<mainindex<<std::endl;
			}
		}

		//-----------------------------------------------------------------
		// Recalculate the centroid and store in the seed values
		//-----------------------------------------------------------------
		sigmal.assign(m_nSuperpixels, 0);
		sigmaa.assign(m_nSuperpixels, 0);
		sigmab.assign(m_nSuperpixels, 0);
		sigmax.assign(m_nSuperpixels, 0);
		sigmay.assign(m_nSuperpixels, 0);
		clustersize.assign(m_nSuperpixels, 0);

		for( int j = 0; j < sz; j++ )
		{
			int temp = labels[j];
			//std::cout<<j<<":"<<labels[j]<<std::endl;
			sigmal[labels[j]] += m_rvec[j];
			sigmaa[labels[j]] += m_gvec[j];
			sigmab[labels[j]] += m_bvec[j];
			sigmax[labels[j]] += (j%m_width);
			sigmay[labels[j]] += (j/m_width);

			clustersize[labels[j]]++;
		}

		{for( int k = 0; k < num; k++ )
		{
			//_ASSERT(clustersize[k] > 0);
			if( clustersize[k] <= 0 ) clustersize[k] = 1;
			inv[k] = 1.0/double(clustersize[k]);//computing inverse now to multiply, than divide later
		}}
		
		{for( int k = 0; k < num; k++ )
		{
			kseedsr[k] = sigmal[k]*inv[k];
			kseedsg[k] = sigmaa[k]*inv[k];
			kseedsb[k] = sigmab[k]*inv[k];
			kseedsx[k] = sigmax[k]*inv[k];
			kseedsy[k] = sigmay[k]*inv[k];
		}}
	}
}
bool isNeighbor(int x, int y, int wx, int wy)
{
	if (abs(x-wx) ==1 && abs(y-wy) == 0 ||
		abs(y-wy) == 1 && abs(x-wx) == 0)
		return true;
	else
		return false;
}
void ComSuperpixel::SuperpixelLattice(uchar4 * rgbBuffer,unsigned width, unsigned height, int step, float alpha,int& num,int* labels,SLICClusterCenter* centers)
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
		m_rvec[j] = rgbBuffer[j].x;
		m_gvec[j] = rgbBuffer[j].y;
		m_bvec[j] = rgbBuffer[j].z;

	}

	vector<double> edgemag(0);
	DetectRGBEdges(m_rvec, m_gvec, m_bvec, m_width, m_height, edgemag);
	GetRGBXYSeeds_ForGivenStep(kseedsr,kseedsg,kseedsb,kseedsx,kseedsy,step,true,edgemag);
	num = m_nSuperpixels;
	//iteration
	vector<double> sigmal(m_nSuperpixels, 0);
	vector<double> sigmaa(m_nSuperpixels, 0);
	vector<double> sigmab(m_nSuperpixels, 0);
	vector<double> sigmax(m_nSuperpixels, 0);
	vector<double> sigmay(m_nSuperpixels, 0);
	vector<int> clustersize(m_nSuperpixels, 0);
	vector<double> inv(m_nSuperpixels, 0);//to store 1/clustersize[k] values
	int itr = 0;
	const int dx4[4] = {-1,  0,  1, 0,};
	const int dy4[4] = { 0, -1, 0, 1};
	const int dx8[8] = {-1, -1,  0,  1, 1, 1, 0, -1};
	const int dy8[8] = { 0, -1, -1, -1, 0, 1, 1,  1};
	int nlx[4];
	int nly[4];
	int nx[8],ny[8];
	int nnx[8],nny[8];
	while(itr < 10)
	{
		itr++;
		//std::cout<<itr<<"------------"<<std::endl;
		vector<bool> istaken(sz, false);

		int mainindex(0);
		for( int j = 0; j < m_height; j++ )
		{
			for( int k = 0; k < m_width; k++ )
			{
				
				//std::cout<<k<<" , "<<j<<std::endl;
				int np(0);
				std::vector<int> nl;
				
				mainindex = k+j*m_width;
				/*if (k==304 && j==465)
					std::cout<<labels[mainindex]<<std::endl;*/
				for( int i = 0; i < 4; i++ )
				{
					int x = k + dx4[i];
					int y = j + dy4[i];

					if( (x >= 0 && x < m_width) && (y >= 0 && y < m_height) )
					{
						int index = y*m_width + x;
						if( labels[mainindex] != labels[index] ) 
						{
							
							nlx[np] = x;
							nly[np] = y;
							np++;
							nl.push_back(labels[index]);
						}											
					}
				}
				if( np > 0 )//change to 2 or 3 for thinner lines
				{
					double min = Distance(k,j,labels[mainindex],kseedsr,kseedsb,kseedsb,kseedsx,kseedsy);
					int idx = -1;
					for(int i=0; i<nl.size(); i++)
					{
						double dis = Distance(k,j,nl[i],kseedsr,kseedsb,kseedsb,kseedsx,kseedsy);
						if (dis < min)
						{
							min = dis;
							idx = i;
						}
					}
					if (idx >=0)
					{
						
						//检查约束1，p所在8邻域内与p的label相同的像素必须是联通的
						bool pass1(true);
						
						int nc(0);
						int nnc(0);
						for(int i=0; i<8; i++)
						{
							int x = k + dx8[i];
							int y = j + dy8[i];
							if ( (x >= 0 && x < m_width) && (y >= 0 && y < m_height) )
							{
								nnx[nnc] = x;
								nny[nnc] = y;
								nnc++;
								int index = y*m_width + x;
								if( labels[mainindex] == labels[index] ) 
								{				
									nx[nc] = x;
									ny[nc] = y;
									nc++;
								}
							}
						}
						if (nc>1)
						{
							bool xflag(true),yflag(true);
							for(int i=0; i<nc; i++)
							{
								if (nx[i] == k)
										xflag = false;
								if (ny[i] == j)
										yflag  = false;
								bool connect(false);
								for(int j=0; j<nc; j++)
								{
									if (j!=i && (abs(nx[i]-nx[j])<2 && abs(ny[i] - ny[j]) <2))
									{
										connect = true;
										break;
									}

								}
								if (!connect)
								{
									pass1 = false;
									break;
								}
							}
							if (pass1 && (xflag || yflag))
								pass1 = false;
						}
						
						//约束2，对于p四邻域内的邻居label n，在p的8邻域内必须存在除p之外的点是n的邻居
						bool pass2(true);
						for(int i=0; i<np; i++)
						{
							if (i!=idx)
							{
								bool connect(false);
								int x = nnx[i];
								int y = nny[i];
								for(int j=0; j<nnc; j++)
								{
									if (isNeighbor(x,y,nnx[j],nny[j]))
									{
										connect = true;
										break;
									}
								}
								if (!connect)
								{
									pass2 = false;
									break;
								}
							}
						}
						if (pass2 && pass1 )
							labels[mainindex] = nl[idx];
						//if (k==304 && j==465)
						//{
						//	std::cout<<labels[mainindex]<<" "<<nl[idx]<<std::endl;
						//	for(int i=0; i<8; i++)
						//	{
						//		int x = k + dx8[i];
						//		int y = j + dy8[i];
						//		if ( (x >= 0 && x < m_width) && (y >= 0 && y < m_height) )
						//		{
						//			int index = y*m_width + x;
						//			std::cout<< labels[index]<<" ";
						//		}
						//	}
						//	std::cout<<std::endl;

						//}

					}
				}
				
				//std::cout<<mainindex<<std::endl;
			}
		}

		//-----------------------------------------------------------------
		// Recalculate the centroid and store in the seed values
		//-----------------------------------------------------------------
		sigmal.assign(m_nSuperpixels, 0);
		sigmaa.assign(m_nSuperpixels, 0);
		sigmab.assign(m_nSuperpixels, 0);
		sigmax.assign(m_nSuperpixels, 0);
		sigmay.assign(m_nSuperpixels, 0);
		clustersize.assign(m_nSuperpixels, 0);

		for( int j = 0; j < sz; j++ )
		{
			int temp = labels[j];
			//std::cout<<j<<":"<<labels[j]<<std::endl;
			sigmal[labels[j]] += m_rvec[j];
			sigmaa[labels[j]] += m_gvec[j];
			sigmab[labels[j]] += m_bvec[j];
			sigmax[labels[j]] += (j%m_width);
			sigmay[labels[j]] += (j/m_width);

			clustersize[labels[j]]++;
		}

		for( int k = 0; k < num; k++ )
		{
			//_ASSERT(clustersize[k] > 0);
			if( clustersize[k] <= 0 ) clustersize[k] = 1;
			inv[k] = 1.0/double(clustersize[k]);//computing inverse now to multiply, than divide later
		}
		
		for( int k = 0; k < num; k++ )
		{
			kseedsr[k] = sigmal[k]*inv[k];
			kseedsg[k] = sigmaa[k]*inv[k];
			kseedsb[k] = sigmab[k]*inv[k];
			kseedsx[k] = sigmax[k]*inv[k];
			kseedsy[k] = sigmay[k]*inv[k];
		}
	}
	for(int i=0; i<num; i++)
	{
		centers[i].xy = make_float2(kseedsx[i],kseedsy[i]);
		centers[i].rgb = make_float4(kseedsr[i],kseedsg[i],kseedsb[i],0);
		centers[i].nPoints = step*step;
	}
}
void ComSuperpixel::SuperpixelLattice(unsigned int * rgbBuffer,unsigned width, unsigned height, int step, float alpha,int& num, int* labels)
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
	GetRGBXYSeeds_ForGivenStep(kseedsr,kseedsg,kseedsb,kseedsx,kseedsy,step,true,edgemag);
	num = m_nSuperpixels;
	//iteration
	vector<double> sigmal(m_nSuperpixels, 0);
	vector<double> sigmaa(m_nSuperpixels, 0);
	vector<double> sigmab(m_nSuperpixels, 0);
	vector<double> sigmax(m_nSuperpixels, 0);
	vector<double> sigmay(m_nSuperpixels, 0);
	vector<int> clustersize(m_nSuperpixels, 0);
	vector<double> inv(m_nSuperpixels, 0);//to store 1/clustersize[k] values
	int itr = 0;
	const int dx4[4] = {-1,  0,  1, 0,};
	const int dy4[4] = { 0, -1, 0, 1};
	const int dx8[8] = {-1, -1,  0,  1, 1, 1, 0, -1};
	const int dy8[8] = { 0, -1, -1, -1, 0, 1, 1,  1};
	int nlx[4];
	int nly[4];
	int nx[8],ny[8];
	int nnx[8],nny[8];
	while(itr < 10)
	{
		itr++;
		//std::cout<<itr<<"------------"<<std::endl;
		vector<bool> istaken(sz, false);

		int mainindex(0);
		for( int j = 0; j < m_height; j++ )
		{
			for( int k = 0; k < m_width; k++ )
			{
				
				//std::cout<<k<<" , "<<j<<std::endl;
				int np(0);
				std::vector<int> nl;
				
				mainindex = k+j*m_width;
				/*if (k==304 && j==465)
					std::cout<<labels[mainindex]<<std::endl;*/
				for( int i = 0; i < 4; i++ )
				{
					int x = k + dx4[i];
					int y = j + dy4[i];

					if( (x >= 0 && x < m_width) && (y >= 0 && y < m_height) )
					{
						int index = y*m_width + x;
						if( labels[mainindex] != labels[index] ) 
						{
							
							nlx[np] = x;
							nly[np] = y;
							np++;
							nl.push_back(labels[index]);
						}											
					}
				}
				if( np > 0 )//change to 2 or 3 for thinner lines
				{
					double min = Distance(k,j,labels[mainindex],kseedsr,kseedsb,kseedsb,kseedsx,kseedsy);
					int idx = -1;
					for(int i=0; i<nl.size(); i++)
					{
						double dis = Distance(k,j,nl[i],kseedsr,kseedsb,kseedsb,kseedsx,kseedsy);
						if (dis < min)
						{
							min = dis;
							idx = i;
						}
					}
					if (idx >=0)
					{
						
						//检查约束1，p所在8邻域内与p的label相同的像素必须是联通的
						bool pass1(true);
						
						int nc(0);
						int nnc(0);
						for(int i=0; i<8; i++)
						{
							int x = k + dx8[i];
							int y = j + dy8[i];
							if ( (x >= 0 && x < m_width) && (y >= 0 && y < m_height) )
							{
								nnx[nnc] = x;
								nny[nnc] = y;
								nnc++;
								int index = y*m_width + x;
								if( labels[mainindex] == labels[index] ) 
								{				
									nx[nc] = x;
									ny[nc] = y;
									nc++;
								}
							}
						}
						if (nc>1)
						{
							bool xflag(true),yflag(true);
							for(int i=0; i<nc; i++)
							{
								if (nx[i] == k)
										xflag = false;
								if (ny[i] == j)
										yflag  = false;
								bool connect(false);
								for(int j=0; j<nc; j++)
								{
									if (j!=i && (abs(nx[i]-nx[j])<2 && abs(ny[i] - ny[j]) <2))
									{
										connect = true;
										break;
									}

								}
								if (!connect)
								{
									pass1 = false;
									break;
								}
							}
							if (pass1 && (xflag || yflag))
								pass1 = false;
						}
						
						//约束2，对于p四邻域内的邻居label n，在p的8邻域内必须存在除p之外的点是n的邻居
						bool pass2(true);
						for(int i=0; i<np; i++)
						{
							if (i!=idx)
							{
								bool connect(false);
								int x = nnx[i];
								int y = nny[i];
								for(int j=0; j<nnc; j++)
								{
									if (isNeighbor(x,y,nnx[j],nny[j]))
									{
										connect = true;
										break;
									}
								}
								if (!connect)
								{
									pass2 = false;
									break;
								}
							}
						}
						if (pass2 && pass1 )
							labels[mainindex] = nl[idx];
						//if (k==304 && j==465)
						//{
						//	std::cout<<labels[mainindex]<<" "<<nl[idx]<<std::endl;
						//	for(int i=0; i<8; i++)
						//	{
						//		int x = k + dx8[i];
						//		int y = j + dy8[i];
						//		if ( (x >= 0 && x < m_width) && (y >= 0 && y < m_height) )
						//		{
						//			int index = y*m_width + x;
						//			std::cout<< labels[index]<<" ";
						//		}
						//	}
						//	std::cout<<std::endl;

						//}

					}
				}
				
				//std::cout<<mainindex<<std::endl;
			}
		}

		//-----------------------------------------------------------------
		// Recalculate the centroid and store in the seed values
		//-----------------------------------------------------------------
		sigmal.assign(m_nSuperpixels, 0);
		sigmaa.assign(m_nSuperpixels, 0);
		sigmab.assign(m_nSuperpixels, 0);
		sigmax.assign(m_nSuperpixels, 0);
		sigmay.assign(m_nSuperpixels, 0);
		clustersize.assign(m_nSuperpixels, 0);

		for( int j = 0; j < sz; j++ )
		{
			int temp = labels[j];
			//std::cout<<j<<":"<<labels[j]<<std::endl;
			sigmal[labels[j]] += m_rvec[j];
			sigmaa[labels[j]] += m_gvec[j];
			sigmab[labels[j]] += m_bvec[j];
			sigmax[labels[j]] += (j%m_width);
			sigmay[labels[j]] += (j/m_width);

			clustersize[labels[j]]++;
		}

		{for( int k = 0; k < num; k++ )
		{
			//_ASSERT(clustersize[k] > 0);
			if( clustersize[k] <= 0 ) clustersize[k] = 1;
			inv[k] = 1.0/double(clustersize[k]);//computing inverse now to multiply, than divide later
		}}
		
		{for( int k = 0; k < num; k++ )
		{
			kseedsr[k] = sigmal[k]*inv[k];
			kseedsg[k] = sigmaa[k]*inv[k];
			kseedsb[k] = sigmab[k]*inv[k];
			kseedsx[k] = sigmax[k]*inv[k];
			kseedsy[k] = sigmay[k]*inv[k];
		}}
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
void ComSuperpixel::GetRGBXYSeeds_ForGivenStep(
	vector<double>&				kseedsl,
	vector<double>&				kseedsa,
	vector<double>&				kseedsb,
	vector<double>&				kseedsx,
	vector<double>&				kseedsy,
	const int&					Step,
	const bool&					perturbseeds,
	const vector<double>&		edgemag)
{
	int sz = m_width*m_height;
	
	m_radius = Step/2;
	int T = Step;
	int xoff = T/2;
	int yoff = T/2;

	int n(0);int r(0);
	for( int y = 0; y < (m_height+T-1)/T; y++ )
	{
		for( int x = 0; x < (m_width+T-1)/T; x++ )
		{
			int X = x*T + xoff;//square grid
			int Y = y*T + yoff;
			int i = Y*m_width + X;
			kseedsx.push_back(X);
			kseedsy.push_back(Y);
			double avgR(0);
			double avgG(0);
			double avgB(0);
			int count(0);
			for(int k= X - xoff; k<= X + xoff; k++)
			{
				if ( k > m_width-1)
					continue;
				for(int j=Y-yoff; j<= Y+yoff; j++)
				{
					if ( j > m_height-1 )
						continue;

					int idx  = k + j*m_width;
					avgR += m_rvec[idx];
					avgG += m_gvec[idx];
					avgB += m_bvec[idx];
					m_labels[idx] = n;
					count++;
					
				}
			}
			kseedsl.push_back(avgR/count);
			kseedsa.push_back(avgG/count);
			kseedsb.push_back(avgB/count);
			n++;
		}		
	}
	m_nSuperpixels = ((m_height+T-1)/T)*((m_width+T-1)/T);
	if(perturbseeds)
	{
		PerturbSeeds(kseedsl, kseedsa, kseedsb, kseedsx, kseedsy, edgemag);
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
	m_radius = step/2;
	int T = step;
	int xoff = T/2;
	int yoff = T/2;

	int n(0);int r(0);
	for( int y = 0; y <= m_height/T; y++ )
	{
		int Y = y*T + yoff;
		if( Y > m_height-1 )
		{
			Y = (y*T + m_height-1)/2;
		}

		for( int x = 0; x <= m_width/T; x++ )
		{
			int X = x*T + xoff;//square grid
			//int X = x*step + (xoff<<(r&0x1));//hex grid
			if(X > m_width-1)
			{
				X = (x*T + m_width-1)/2;
			}

			int i = Y*m_width + X;

			//_ASSERT(n < K);

			//kseedsl[n] = m_lvec[i];
			//kseedsa[n] = m_avec[i];
			//kseedsb[n] = m_bvec[i];
			//kseedsx[n] = X;
			//kseedsy[n] = Y;
			
			kseedsx.push_back(X);
			kseedsy.push_back(Y);
			double avgR(0);
			double avgG(0);
			double avgB(0);
			int count(0);
			for(int k= X - xoff; k<= X + xoff; k++)
			{
				if (k>m_width-1)
					continue;
				for(int j=Y-yoff; j<= Y+yoff; j++)
				{
					if (j>m_height-1)
						continue;
					int idx  = k + j*m_width;
					avgR += m_rvec[idx];
					avgG += m_gvec[idx];
					avgB += m_bvec[idx];
					m_labels[idx] = n;
					count++;
				}
			}
			kseedsl.push_back(avgR/count);
			kseedsa.push_back(avgG/count);
			kseedsb.push_back(avgB/count);
			n++;
		}
		//r++;
	}
	m_nSuperpixels = n;
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