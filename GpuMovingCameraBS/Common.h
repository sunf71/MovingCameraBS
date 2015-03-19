#pragma once
#include <io.h>
#include <direct.h>
#include <vector>
#include <string>

static inline int CreateDir(char *pszDir)
{
	int i = 0;
	int iRet;
	int iLen = strlen(pszDir);
	//��ĩβ��/
	if (pszDir[iLen - 1] != '\\' && pszDir[iLen - 1] != '/')
	{
		pszDir[iLen] = '/';
		pszDir[iLen + 1] = '\0';
	}

	// ����Ŀ¼
	for (i = 0;i < iLen;i ++)
	{
		if (pszDir[i] == '\\' || pszDir[i] == '/')
		{ 
			pszDir[i] = '\0';

			//���������,����
			iRet = _access(pszDir,0);
			if (iRet != 0)
			{
				iRet = _mkdir(pszDir);
				if (iRet != 0)
				{
					return -1;
				} 
			}
			//֧��linux,������\����/
			pszDir[i] = '/';
		} 
	}

	return 0;
}
template<typename T>
void mySwap(T*& a, T*& b)
{
	T* tmp = b;
	b = a;
	a = tmp;
}
template<typename T> void safe_delete_array(T*& a) 
{

	if (a!=NULL)
	{
		delete[] a;
		a = NULL;
	}
}

template<typename T> void safe_delete(T*& a) 
{

	if (a!=NULL)
	{
		delete a;
		a = NULL;
	}
}

void DrawHistogram(std::vector<float>& histogram, int size, const std::string name = "histogram");

