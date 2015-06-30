
#pragma once

#define		BLOCK_DIM		256
#define		BLOCK_NUM_MAX	512
//extern __shared__ int s_array[ ];

__global__ 
void error_detection_kernel( char* codeword, int* powAlpha, int* SCache, int i, int MAXN, int n )
{
	int j = blockIdx.x * blockDim.x + threadIdx.x ;

	if( j >= n )
		return;

	__shared__ int	s_powAlpha[BLOCK_DIM] ;
	__shared__ char	s_codeword[BLOCK_DIM] ;

	s_codeword[ threadIdx.x ] = codeword[ j ];
	if(s_codeword[ threadIdx.x ])
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