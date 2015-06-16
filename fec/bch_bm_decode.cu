

#include "bch_bm_decode.cuh"
#include "bch_bm_decode_kernel.cuh"
#include <cuda_runtime.h>

bch_gpu::bch_gpu()
{

}

bch_gpu::~bch_gpu()
{

}


void bch_gpu::initialize(	int *powAlpha, int *indexAlpha, int mNormal, 
							int *S, int nS, 
							int n, int tCapacity, int MAXN )
{
	m_nAlphaSize = 1<<mNormal;
	m_nSSize = nS;
	this->n = n;
	this->tCapacity = tCapacity;

	cudaMalloc( (void**)&d_powAlpha, m_nAlphaSize*sizeof(int) );
	cudaMalloc( (void**)&d_indexAlpha, m_nAlphaSize*sizeof(int) );
	cudaMalloc( (void**)&d_S, m_nSSize*sizeof(int) );

	cudaMemcpy( d_powAlpha, d_powAlpha, m_nAlphaSize*sizeof(int), cudaMemcpyHostToDevice );
	cudaMemcpy( d_indexAlpha, d_indexAlpha, m_nAlphaSize*sizeof(int), cudaMemcpyHostToDevice );
	cudaMemcpy( d_S, S, nS*sizeof(int), cudaMemcpyHostToDevice );

	cudaMalloc( (void**)&d_codeword, n*sizeof(int) );
	
	this->powAlpha = powAlpha;
	this->indexAlpha = indexAlpha;
	this->S = S;
	this->MAXN = MAXN;
}

void bch_gpu::release()
{
	cudaFree( d_powAlpha );
	cudaFree( d_indexAlpha );
	cudaFree( d_S );

	cudaFree( d_codeword );
}


bool bch_gpu::error_detection( char* codeword )
{
	this->codeword = codeword;

	bool syn = false;
	for(int i = 0; i < tCapacity*2; i++)
	{
		S[i] = 0;
		for(int j = 0; j < n; j++)
		{
			if(codeword[j])
				S[i] ^= powAlpha[((i+1)*j)%MAXN];
		}

		S[i] = indexAlpha[S[i]];

		if(S[i] != -1)
			syn = true;

	}

	return syn;
}
