

#include "bch_bm_decode.cuh"
#include "bch_bm_decode_kernel.cuh"
#include <cuda_runtime.h>

#include <vector>
#include <stdlib.h>
#include <iostream>
using namespace std;

#include "dvbUtility.h"

#if USE_TEXTURE_ADDRESS
	cudaArray* arr_alpha;
	cudaChannelFormatDesc channelDesc;
#endif

bch_gpu::bch_gpu()
{

}

bch_gpu::~bch_gpu()
{

}


void bch_gpu::initialize(	int *powAlpha, int *indexAlpha, int mNormal, 
							int *S, int nS, 
							int n, int tCapacity, int MAXN, int tMax )
{
	m_nAlphaSize = 1<<mNormal;
	m_nSSize = nS;
	this->n = n;
	this->tCapacity = tCapacity;
	this->tMax = tMax;

	cudaMalloc( (void**)&d_powAlpha, m_nAlphaSize*sizeof(int) );
	cudaMalloc( (void**)&d_indexAlpha, m_nAlphaSize*sizeof(int) );
	cudaMalloc( (void**)&d_S, m_nSSize*sizeof(int) );

	cudaMemcpy( d_powAlpha, powAlpha, m_nAlphaSize*sizeof(int), cudaMemcpyHostToDevice );
	cudaMemcpy( d_indexAlpha, indexAlpha, m_nAlphaSize*sizeof(int), cudaMemcpyHostToDevice );
	cudaMemcpy( d_S, S, nS*sizeof(int), cudaMemcpyHostToDevice );

	cudaMalloc( (void**)&d_codeword, n*sizeof(char) );
	
	
	cudaMalloc( (void**)&d_SCache, tCapacity*2*BLOCK_DIM*sizeof(int) );
	cudaMemset( d_SCache, 0, tCapacity*2*BLOCK_DIM*sizeof(int) );

	cudaMalloc( (void**)&d_lambda, tCapacity*2*sizeof(int));
	
	cudaMalloc( (void**)&d_el, tMax*2*sizeof(int));
	cudaMemset( d_el, -1, tMax*2*sizeof(int) );
	
	cudaMalloc( (void**)&d_kk, 1*sizeof(int));
	cudaMemset( d_kk, 0, 1*sizeof(int) );

	m_SCache = (int*) calloc(tCapacity*2*BLOCK_NUM_MAX,sizeof(int));

	this->powAlpha = powAlpha;
	this->indexAlpha = indexAlpha;
	this->S = S;
	this->MAXN = MAXN;

	
#if USE_TEXTURE_ADDRESS
	// cuda texture ------------------------------------------------------------------------------------------
	channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindSigned);
    cudaError_t err = cudaMallocArray(&arr_alpha, &channelDesc, m_nAlphaSize, 1);
    cudaMemcpyToArray(arr_alpha, 0, 0, d_powAlpha, m_nAlphaSize * sizeof(int), cudaMemcpyDeviceToDevice);

	texAlpha.addressMode[0] = cudaAddressModeClamp;
    texAlpha.filterMode = cudaFilterModePoint;
    texAlpha.normalized = false;

	cudaBindTextureToArray(texAlpha, arr_alpha, channelDesc);

#endif
}

void bch_gpu::release()
{
	cudaFree( d_powAlpha );
	cudaFree( d_indexAlpha );
	cudaFree( d_S );
	cudaFree( d_SCache );

	cudaFree( d_codeword );

	cudaFree( d_lambda ); 
	cudaFree( d_el );
	cudaFree( d_kk );

#if USE_TEXTURE_ADDRESS
	//cudaFreeArray( arr_alpha );
#endif
}


bool bch_gpu::error_detection( char* codeword )
{
	this->codeword = codeword;
	cudaMemcpy( d_codeword, codeword, n*sizeof(char), cudaMemcpyHostToDevice );

	dim3 block(BLOCK_DIM);
	dim3 grid( (n+BLOCK_DIM-1)/BLOCK_DIM );

#if 1

	error_detection_kernel<<< grid, block >>>( d_codeword, d_powAlpha, d_SCache, char(tCapacity*2), MAXN, n );

#else
	for(int i = 0; i < tCapacity*2; i++)
	{
		error_detection_kernel<<< grid, block >>>( d_codeword, d_powAlpha, d_SCache, i, MAXN, n );
	}
#endif

	cudaMemcpy( m_SCache, d_SCache, tCapacity*2*grid.x * sizeof(int), cudaMemcpyDeviceToHost );
		
		
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

	for(int i = 0; i < tCapacity*2; i++)
	{
		S[i] = 0;
		for( int j=0; j< grid.x; j++ )
		{
			S[i] ^= m_SCache[j+i*grid.x];
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

void bch_gpu::chienSearch( int* lambda, int* el, int L )
{

	// 0.6 ms 
	cudaMemcpy( d_lambda, lambda, tCapacity * 2 * sizeof(int), cudaMemcpyHostToDevice );

	cudaMemset( d_el, -1, tMax*2*sizeof(int) );
	cudaMemset( d_kk, 0, 1*sizeof(int) );

	dim3 block( BLOCK_DIM );
	dim3 grid( (MAXN + BLOCK_DIM - 1)/BLOCK_DIM );
	chien_search_kernel<<< grid, block >>>( d_powAlpha, d_lambda, d_el, d_kk, L, MAXN );

	cudaMemcpy( el, d_el, tMax * 2 * sizeof(int), cudaMemcpyDeviceToHost );

#if WRITE_FILE_FOR_DRIVER
	static bool bRunOnce1 = false;
	if( !bRunOnce1 ){
		std::vector<int> paramSize;
		paramSize.push_back( tCapacity );
		paramSize.push_back( m_nAlphaSize );
		paramSize.push_back( tMax );
		paramSize.push_back( MAXN );
		paramSize.push_back( L );
		//writeFile( tCapacity, m_nAlpha, tMax, MAXN, L, "../data/chienSearchSize.txt" );
		writeFile( paramSize, "../data/chienSearchSize.txt" );
		writeArray( lambda, tCapacity * 2, "../data/lambda.txt" );		
		writeArray( powAlpha, m_nAlphaSize, "../data/powAlpha.txt" );
		writeArray( el, tMax * 2, "../data/el.txt" );

		bRunOnce1 = true;
	}
#endif
}
