
#pragma once

class bch_gpu
{
public:
	bch_gpu();
	~bch_gpu();

	void initialize( int *powAlpha, int *indexAlpha, int mNormal, 
					 int *S, int nS,
					 int n, int tCapacity, int MAXN, int tMax );

	void release();

	bool error_detection( char* codeword );

	void chienSearch( int* lambda, int* el, int L );

protected:
	int *d_powAlpha, *d_indexAlpha;
	int *d_S;          // Syndrome vector
	char* d_codeword;

	int *d_SCache; 

	int *d_el, *d_lambda, *d_kk;

	int *m_SCache;        

	int m_nAlphaSize;
	int m_nSSize;

	int *powAlpha, *indexAlpha;
	int *S;
	char* codeword;
	int n, tCapacity;
	int MAXN, tMax;
};