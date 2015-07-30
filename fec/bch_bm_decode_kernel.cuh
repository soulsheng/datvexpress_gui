
#pragma once

#define		BLOCK_DIM		256
#define		BLOCK_NUM_MAX	512
//extern __shared__ int s_array[ ];
#define		USE_TEXTURE_ADDRESS	0

#if USE_TEXTURE_ADDRESS
texture<int, 1, cudaReadModeElementType> texAlpha;
#endif

__global__ 
void error_detection_kernel( char* codeword, int* powAlpha, int* SCache, int i, int MAXN, int n )
{
	int j = blockIdx.x * blockDim.x + threadIdx.x ;

	__shared__ int	s_powAlpha[BLOCK_DIM] ;
	__shared__ char	s_codeword[BLOCK_DIM] ;

	s_codeword[ threadIdx.x ] = codeword[ j ];
	if(s_codeword[ threadIdx.x ] && j<n )
  		s_powAlpha[ threadIdx.x ] = powAlpha[ ((i+1)*j)%MAXN ];
	else
		s_powAlpha[ threadIdx.x ] = 0;

	__syncthreads();

	for( int offset = blockDim.x / 2; offset>=1; offset /= 2 )
	{
		if( threadIdx.x < offset )
				s_powAlpha[ threadIdx.x ] ^= s_powAlpha[ threadIdx.x + offset ];

		__syncthreads();
	}


	if( threadIdx.x == 0 )
		SCache[blockIdx.x+i*gridDim.x] = s_powAlpha[0];
	__syncthreads();
}

__global__ 
void error_detection_kernel( char* codeword, int* powAlpha, int* SCache, char t2, int MAXN, int n )
{
	int j = blockIdx.x * blockDim.x + threadIdx.x ;

	__shared__ int	s_powAlpha[BLOCK_DIM] ;

	char b = codeword[ j ];

	for(int i = 0; i < t2; i++)// empty loop cost 30 us
	{
	
	if( b && j<n )
 #if USE_TEXTURE_ADDRESS
		s_powAlpha[ threadIdx.x ]=  tex1D(texAlpha, ((i+1)*j)%MAXN );
#else 	
		s_powAlpha[ threadIdx.x ] = powAlpha[ ((i+1)*j)%MAXN ];
#endif
	else
		s_powAlpha[ threadIdx.x ] = 0;

	__syncthreads();	// 120 us = 24*5 us

	for( int offset = blockDim.x / 2; offset>=1; offset /= 2 )
	{
		if( threadIdx.x < offset )
				s_powAlpha[ threadIdx.x ] ^= s_powAlpha[ threadIdx.x + offset ];

		__syncthreads();
	}// 140 us = 24*6 us 


	if( threadIdx.x == 0 ) // 1 us
		SCache[blockIdx.x+i*gridDim.x] = s_powAlpha[0];
	__syncthreads();

	}
}

__global__
void chien_search_kernel( int* powAlpha, int* lambda, int* el, int* kk, int L, int MAXN )
{
	int i = blockIdx.x * blockDim.x + threadIdx.x ;

	if( i >= MAXN )
		return;

	int tmp = 0;
	
	for(int j = 1; j <=L; j++)
			tmp ^= powAlpha[(lambda[j]+i*j)%MAXN];

	if (tmp == 1)
	{
		int k = atomicAdd( kk, 1 );
		// roots inversion give the error locations
		el[k] = (MAXN-i)%MAXN;
	}
}