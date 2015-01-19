#pragma once
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