
#pragma once

#define		BLOCK_DIM		256
#define		BLOCK_NUM_MAX	512
//extern __shared__ int s_array[ ];

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
		SCache[blockIdx.x] = s_powAlpha[0];
	__syncthreads();
}

__global__
void chien_search_kernel( int* powAlpha, int* lambda, int* el, int* kk, int L, int MAXN )
{
	int i = blockIdx.x * blockDim.x + threadIdx.x ;

	if( i >= MAXN )
		return;

	for(int j = 1; j <=L; j++)
			tmp ^= powAlpha[(lambda[j]+i*j)%MAXN];

	if (tmp == 1)
	{
		int k = atomicAdd( kk, 1 );
		// roots inversion give the error locations
		el[k] = (MAXN-i)%MAXN;
	}
}