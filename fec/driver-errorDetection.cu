
#include "driver-errorDetection.cuh"

#include <cuda_runtime.h>

#include <fstream>
#include <iostream>
using namespace std;


#define		MAX_LOCAL_CACHE		20


#define		SIZE_BLOCK			256
#define		SIZE_BLOCK_2D_X		32

#define		USE_BLOCK_2D		0
#define		N_FRAME				20	// time scales as long as data length scales

#define		BLOCK_DIM		256
#define		BLOCK_NUM_MAX	512
//extern __shared__ int s_array[ ];

__global__ 
void error_detection_kernel( char* n_codeword, int* powAlpha, int* n_SCache, int i, int MAXN, int n, int nFrame )
{
	int j = blockIdx.x * blockDim.x + threadIdx.x ;

	__shared__ int	s_powAlpha[BLOCK_DIM] ;
	__shared__ char	s_codeword[BLOCK_DIM] ;

	for( int frame = 0; frame < nFrame; frame ++ )	{

	const char	*codeword	= n_codeword + frame * n;
	int			*SCache		= n_SCache + frame * gridDim.x;

	s_codeword[ threadIdx.x ] = codeword[ j ];
	if(s_codeword[ threadIdx.x ] && j<n)
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
}


bool driverErrorDetection::launch()
{
#if USE_BLOCK_2D
	
	dim3 block( SIZE_BLOCK_2D_X, MAX_VAR_NODE );
	dim3 grid;
	grid.x = (nvar * MAX_VAR_NODE + SIZE_BLOCK_2D_X * MAX_VAR_NODE - 1) 
				/ (SIZE_BLOCK_2D_X * MAX_VAR_NODE) ;

	updateVariableNodeOpti2D_kernel<<< grid, block >>>( nvar, ncheck, 
		d_sumX1, d_mcv, d_iind, d_input, 
		d_output, d_mvc );
#else

	dim3 block(BLOCK_DIM);
	dim3 grid( (m_nCodeword+BLOCK_DIM-1)/BLOCK_DIM );

	error_detection_kernel<<< grid, block >>>( d_codeword, d_powAlpha, d_SCache, 0, MAXN, m_nCodeword, N_FRAME );

#endif

	cudaError_t	status = cudaGetLastError();
	return status == cudaSuccess ;
}

bool driverErrorDetection::verify()
{
	cudaMemcpy( SCache, d_SCache, m_nGrid * sizeof(int) * N_FRAME, cudaMemcpyDeviceToHost );

	// output
	int i = 0;
	for ( ; i < m_nGrid * N_FRAME; i++ )
	{
		if ( ref_SCache[i] != SCache[i] )
			break;
	}

	if ( i < m_nGrid * N_FRAME )
		return false;

	return true;
}

template <typename T>
void 	readArray(T* pArray, int nSize, char* strFileName)
{
	FILE* fp = NULL;
	fp = fopen( strFileName, "rb" );
	if(fp == NULL)
	{
		printf("failed to open: %s!\n", strFileName);
	}
	fread( pArray, sizeof(T), nSize, fp);
	fclose(fp);
}

void	readFile(int& nCodeword, int& nAlpha, int& nGrid, int &nMax, char* filename)
{
	FILE* fp;
	fp = fopen( filename, "rb" );
	if( !fp )
		return;

	fread( &nCodeword, sizeof(int), 1, fp );
	fread( &nAlpha, sizeof(int), 1, fp );
	fread( &nGrid, sizeof(int), 1, fp );
	fread( &nMax, sizeof(int), 1, fp );

	fclose( fp );
}

driverErrorDetection::driverErrorDetection( )
{
	ifstream  testfile;
	testfile.open( "../data/bchSize.txt" );
	if ( testfile == NULL )
	{
		cout << "Missing ldpc code parameter file - \"bchSize.txt\" in data path!" << endl ;
		return ;
	}
	else
	{
		cout << "Success to load ldpc code parameter file !" << endl ;
	}
	testfile.close();

	readFile( m_nCodeword, m_nAlpha, m_nGrid, MAXN, "../data/bchSize.txt" );

	codeword = (char*)malloc(m_nCodeword * sizeof(char) * N_FRAME);
	powAlpha = (int*)malloc(m_nAlpha  * sizeof(int));
	SCache = (int*)malloc(m_nGrid * sizeof(int) * N_FRAME);

	ref_SCache = (int*)malloc(m_nGrid * sizeof(int) * N_FRAME);
	
	readArray( codeword, m_nCodeword, "../data/codeword.txt" );		
	readArray( powAlpha, m_nAlpha, "../data/powAlpha.txt" );
	readArray( ref_SCache, m_nGrid, "../data/SCache.txt" );  

	for( int i = 0; i < N_FRAME; i ++ )
	{
		memcpy( codeword + i * m_nCodeword, codeword,  m_nCodeword * sizeof(char) );
		memcpy( ref_SCache + i * m_nGrid, ref_SCache,  m_nGrid * sizeof(int) );
	}

	cudaMalloc( (void**)&d_codeword, m_nCodeword*sizeof(char) * N_FRAME );
	cudaMemcpy( d_codeword, codeword, m_nCodeword*sizeof(char) * N_FRAME, cudaMemcpyHostToDevice );

	cudaMalloc( (void**)&d_powAlpha, m_nAlpha*sizeof(int) );
	cudaMemcpy( d_powAlpha, powAlpha, m_nAlpha * sizeof(int), cudaMemcpyHostToDevice );

	cudaMalloc( (void**)&d_SCache, m_nGrid*sizeof(int) * N_FRAME );
	cudaMemset( d_SCache, 0, m_nGrid*sizeof(int) * N_FRAME );

}

driverErrorDetection::~driverErrorDetection()
{
	// host
	free(powAlpha);
	free(SCache);
	free(codeword);

	free(ref_SCache);

	// device
	cudaFree( d_powAlpha );
	cudaFree( d_SCache );	
	cudaFree( d_codeword );
}