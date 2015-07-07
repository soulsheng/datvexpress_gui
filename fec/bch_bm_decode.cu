

#include "bch_bm_decode.cuh"
#include "bch_bm_decode_kernel.cuh"
#include <cuda_runtime.h>

#include <stdlib.h>


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

	cudaMemcpy( d_powAlpha, powAlpha, m_nAlphaSize*sizeof(int), cudaMemcpyHostToDevice );
	cudaMemcpy( d_indexAlpha, indexAlpha, m_nAlphaSize*sizeof(int), cudaMemcpyHostToDevice );
	cudaMemcpy( d_S, S, nS*sizeof(int), cudaMemcpyHostToDevice );

	cudaMalloc( (void**)&d_codeword, n*sizeof(char) );
	
	
	cudaMalloc( (void**)&d_SCache, BLOCK_DIM*sizeof(int) );
	cudaMemset( d_SCache, 0, BLOCK_DIM*sizeof(int) );

	m_SCache = (int*) calloc(BLOCK_NUM_MAX,sizeof(int));

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
	cudaMemcpy( d_codeword, codeword, n*sizeof(char), cudaMemcpyHostToDevice );

	dim3 block(BLOCK_DIM);
	dim3 grid( (n+BLOCK_DIM-1)/BLOCK_DIM );
	for(int i = 0; i < tCapacity*2; i++)
	{
		error_detection_kernel<<< grid, block >>>( d_codeword, d_powAlpha, d_SCache, i, MAXN, n );
		cudaMemcpy( m_SCache, d_SCache, grid.x * sizeof(int), cudaMemcpyDeviceToHost );
		
		
#if WRITE_FILE_FOR_DRIVER
	static bool bRunOnce1 = false;
	if( !bRunOnce1 ){
		writeFile( n, m_nAlphaSize, grid.x, MAXN, "../data/bchSize.txt" );
		writeArray( codeword, n, "../data/codeword.txt" );		
		writeArray( powAlpha, m_nAlphaSize, "../data/powAlpha.txt" );
		writeArray( m_SCache, grid.x, "../data/SCache.txt" );

		bRunOnce1 = true;
	}
#endif

		S[i] = 0;
		for( int j=0; j< grid.x; j++ )
		{
			S[i] ^= m_SCache[j];
		}
	}
	
	bool syn = false;
	for(int i = 0; i < tCapacity*2; i++)
	{
		S[i] = indexAlpha[S[i]];

		if(S[i] != -1)
			syn = true;

	}

	return syn;
}
