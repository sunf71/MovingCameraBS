#include "RegionObjectness.h"


void RegionOutBorder(int i, std::vector<SPRegion>& regions)
{
	regions[i].outBorderSPs.clear();
	if (i < regions.size() && regions[i].size > 0)
	{
		for (size_t j = 0; j < regions[i].neighbors.size(); j++)
		{
			int nid = regions[i].neighbors[j];
			if (regions[nid].size > 0)
			{
				size_t k = 0;
				for (; k < regions[nid].neighbors.size(); k++)
				{
					if (regions[nid].neighbors[k] == i)
						break;
				}
				if (k < regions[nid].borderSpIndices.size() )
				{
					for (size_t m = 0; m < regions[nid].borderSpIndices[k].size(); m++)
					{
						regions[i].outBorderSPs.push_back(regions[nid].borderSpIndices[k][m]);
					}
				}
				
			}
		}
		
	}
}


void GetRegionOutBorder(std::vector<SPRegion>& regions)
{
	for (size_t i = 0; i < regions.size(); i++)
	{
		if (regions[i].size > 0)
		{
			std::vector<int> borderSPs;
			RegionOutBorder(i, regions);
			
		}
	}
}


float RegionObjectness(std::vector<SPRegion>& regions, int i, SuperpixelComputer* computer, HISTOGRAMS& colorHist, cv::Mat& edgeMap)
{
	return 0;
}