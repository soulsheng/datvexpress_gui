
#include "driver-chienSearch.cuh"

#include <cuda_runtime.h>

#include <fstream>
#include <iostream>
using namespace std;


#define		MAX_LOCAL_CACHE		20


#define		SIZE_BLOCK			256
#define		SIZE_BLOCK_2D_X		32

#define		USE_BLOCK_2D		0
#define		N_FRAME				1	// time scales as long as data length scales

#define		BLOCK_DIM		256
#define		BLOCK_NUM_MAX	512

#define		MAXT 12         // Max corrective capacity

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

bool driverChienSearch::launch()
{

	// 0.6 ms 

	dim3 block( BLOCK_DIM );
	dim3 grid( (MAXN + BLOCK_DIM - 1)/BLOCK_DIM );
	chien_search_kernel<<< grid, block >>>( d_powAlpha, d_lambda, d_el, d_kk, L, MAXN );

	cudaError_t	status = cudaGetLastError();
	return status == cudaSuccess ;
}

bool driverChienSearch::verify()
{
	cudaMemcpy( el, d_el, tMax * 2 * sizeof(int) * N_FRAME, cudaMemcpyDeviceToHost );

	// output
	int i = 0;
	for ( ; i < tMax * 2 * N_FRAME; i++ )
	{
		if ( ref_el[i] != el[i] )
			break;
	}

	if ( i < tMax * 2 * N_FRAME )
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

void	readFile(int& tCapacity, int& nAlpha, int& tMax, int &nMax, int &L, char* filename)
{
	FILE* fp;
	fp = fopen( filename, "rb" );
	if( !fp )
		return;

	fread( &tCapacity, sizeof(int), 1, fp );
	fread( &nAlpha, sizeof(int), 1, fp );
	fread( &tMax, sizeof(int), 1, fp );
	fread( &nMax, sizeof(int), 1, fp );
	fread( &L, sizeof(int), 1, fp );

	fclose( fp );
}

driverChienSearch::driverChienSearch( )
{
	ifstream  testfile;
	testfile.open( "../data/chienSearchSize.txt" );
	if ( testfile == NULL )
	{
		cout << "Missing ldpc code parameter file - \"chienSearchSize.txt\" in data path!" << endl ;
		return ;
	}
	else
	{
		cout << "Success to load ldpc code parameter file !" << endl ;
	}
	testfile.close();

	readFile( tCapacity, m_nAlpha, tMax, MAXN, L, "../data/chienSearchSize.txt" );

	powAlpha = (int*)malloc(m_nAlpha  * sizeof(int));
	lambda = (int*) calloc(tCapacity*2,sizeof(int));
	el = (int*) calloc(MAXT*2,sizeof(int));
	ref_el = (int*) calloc(MAXT*2,sizeof(int));

	
	readArray( lambda, tCapacity*2, "../data/lambda.txt" );		
	readArray( powAlpha, m_nAlpha, "../data/powAlpha.txt" );
	readArray( ref_el, MAXT*2, "../data/el.txt" );		

	for( int i = 0; i < N_FRAME; i ++ )
	{
		memcpy( lambda + i * tCapacity*2, lambda,  tCapacity*2 * sizeof(char) );
		memcpy( ref_el + i * MAXT*2, ref_el,  MAXT*2 * sizeof(int) );
	}

	cudaMalloc( (void**)&d_powAlpha, m_nAlpha*sizeof(int) );
	cudaMemcpy( d_powAlpha, powAlpha, m_nAlpha * sizeof(int), cudaMemcpyHostToDevice );

	
	cudaMalloc( (void**)&d_lambda, tCapacity*2*sizeof(int));
	cudaMemcpy( d_lambda, lambda, tCapacity * 2 * sizeof(int), cudaMemcpyHostToDevice );

	cudaMalloc( (void**)&d_el, tMax*2*sizeof(int));
	cudaMemset( d_el, -1, tMax*2*sizeof(int) );
	
	cudaMalloc( (void**)&d_kk, 1*sizeof(int));
	cudaMemset( d_kk, 0, 1*sizeof(int) );


}

driverChienSearch::~driverChienSearch()
{
	// host
	free(powAlpha);
	free(lambda);
	free(el);

	free(ref_el);

	// device
	cudaFree( d_powAlpha );
	cudaFree( d_lambda );	
	cudaFree( d_el );
	cudaFree( d_kk );
}