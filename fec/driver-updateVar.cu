
#include "driver-updateVar.cuh"

#include <cuda_runtime.h>

#include <fstream>
#include <iostream>
using namespace std;


#define		MAX_LOCAL_CACHE		20


#define		SIZE_BLOCK			256
#define		SIZE_BLOCK_2D_X		32

#define		USE_BLOCK_2D		0
#define		N_FRAME				10	// time scales as long as data length scales

__global__ 
void updateVariableNodeOpti_kernel( const int nvar, const int ncheck, const int* sumX1, const int* n_mcv, const int* iind, const int * n_LLRin, 
	char * n_LLRout, int* n_mvc, 
	int nmaxX1, int nmaxX2, int nFrame ) // not used, just for testing performance bound
{	//	mcv const(input)-> mvc (output)
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	
	if( i>= nvar )
		return;
	
	for( int frame = 0; frame < nFrame; frame ++ )	{

	const int	*LLRin	= n_LLRin + frame * nvar;
	const int	*mcv	= n_mcv + frame * ncheck * nmaxX2;
	char	*LLRout = n_LLRout + frame * nvar;
	int		*mvc	= n_mvc + frame * nvar * nmaxX1;

	int mvc_temp = LLRin[i];

	int m[MAX_LOCAL_CACHE];

	for (int jp = 0; jp < sumX1[i]; jp++)
		m[jp] = mcv[ iind[i + jp*nvar] ];


	for (int jp = 0; jp < sumX1[i]; jp++)
		mvc_temp += m[jp];
	

	LLRout[i] = mvc_temp<0;
	
	for (int jp = 0; jp < sumX1[i]; jp++)
			mvc[i + jp*nvar] = mvc_temp - m[jp];
	}
}


__global__ 
void updateVariableNodeOpti2D_kernel( const int nvar, const int ncheck, const int* sumX1, const int* mcv, const int* iind, const int * LLRin, 
	char * LLRout, int* mvc ) // not used, just for testing performance bound
{	//	mcv const(input)-> mvc (output)
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	
	if( i>= nvar )
		return;
		
	__shared__ int mvc_temp[SIZE_BLOCK_2D_X];
	__shared__ int m[MAX_LOCAL_CACHE][SIZE_BLOCK_2D_X];
	

	if( threadIdx.y < sumX1[i] )
		m[threadIdx.y][threadIdx.x] = mcv[ iind[i + threadIdx.y*nvar] ];
	__syncthreads();

	if( threadIdx.y == 0 )
	{
		mvc_temp[threadIdx.x] = LLRin[i];

		for (int jp = 0; jp < sumX1[i]; jp++)
			mvc_temp[threadIdx.x] += m[jp][threadIdx.x];

		LLRout[i] = mvc_temp[threadIdx.x]<0;
	}
	__syncthreads();

	if( threadIdx.y < sumX1[i] )
		mvc[i + threadIdx.y*nvar] = mvc_temp[threadIdx.x] - m[threadIdx.y][threadIdx.x];
	__syncthreads();

}


bool driverUpdataVar::launch()
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

	dim3 block( SIZE_BLOCK );
	dim3 grid( (nvar + block.x - 1) / block.x );

	updateVariableNodeOpti_kernel<<< grid, block >>>( nvar, ncheck, 
		d_sumX1, d_mcv, d_iind, d_input, 
		d_output, d_mvc, 
		nmaxX1, nmaxX2, N_FRAME );

#endif

	cudaError_t	status = cudaGetLastError();
	return status == cudaSuccess ;
}

bool driverUpdataVar::verify()
{
	cudaMemcpy( mvc, d_mvc, nvar * nmaxX1 * sizeof(int) * N_FRAME, cudaMemcpyDeviceToHost );
	cudaMemcpy( output, d_output, nvar * sizeof(char) * N_FRAME, cudaMemcpyDeviceToHost );

	// mvc
	int i = 0;
	for ( ; i < nvar * nmaxX1 * N_FRAME; i++ )
	{
		if ( ref_mvc[i] != mvc[i] )
			break;
	}

	if ( i < nvar * nmaxX1 * N_FRAME )
		return false;

	// output
	int j = 0;
	for ( ; j < nvar * N_FRAME; j++ )
	{
		if ( ref_output[j] != output[j] )
			break;
	}

	if ( j < nvar * N_FRAME )
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

void	readFile(int& nvar, int& ncheck, int& nmaxX1, int& nmaxX2, char* filename)
{
	FILE* fp;
	fp = fopen( filename, "rb" );
	if( !fp )
		return;

	fread( &nvar, sizeof(int), 1, fp );
	fread( &ncheck, sizeof(int), 1, fp );
	fread( &nmaxX1, sizeof(int), 1, fp );
	fread( &nmaxX2, sizeof(int), 1, fp );

	fclose( fp );
}

driverUpdataVar::driverUpdataVar( )
{
	readFile( nvar, ncheck, nmaxX1, nmaxX2, "../data/ldpcSize.txt" );

	sumX1 = (int*)malloc(nvar * sizeof(int));
	iind = (int*)malloc(nvar * nmaxX1 * sizeof(int));
	mvc = (int*)malloc(nvar * nmaxX1 * sizeof(int) * N_FRAME);
	mcv = (int*)malloc(ncheck * nmaxX2 * sizeof(int) * N_FRAME);
	input = (int*)malloc(nvar * sizeof(int) * N_FRAME);
	output = (char*)malloc(nvar * sizeof(char) * N_FRAME);

	ref_mvc = (int*)malloc(nvar * nmaxX1 * sizeof(int) * N_FRAME);
	ref_output = (char*)malloc(nvar * sizeof(char) * N_FRAME);
	
	ifstream  testfile;
	testfile.open( "../data/sumX1.txt" );
	if ( testfile == NULL )
	{
		cout << "Missing ldpc code parameter file - \"sumX1.txt\" in data path!" << endl ;
		return ;
	}
	else
	{
		cout << "Success to load ldpc code parameter file !" << endl ;
	}
	testfile.close();

	readArray( sumX1, nvar, "../data/sumX1.txt" );

	readArray( iind, nvar * nmaxX1, "../data/iind.txt" );

	readArray( ref_output, nvar, "../data/output.txt" );

	readArray( input, nvar, "../data/input.txt" );

	readArray( mcv, ncheck * nmaxX2, "../data/mcv.txt" );	

	readArray( ref_mvc, nvar * nmaxX1, "../data/mvc.txt" );		

	for( int i = 0; i < N_FRAME; i ++ )
	{
		memcpy( ref_output + i * nvar, ref_output,  nvar * sizeof(char) );
		memcpy( input + i * nvar, input,  nvar * sizeof(int) );
		memcpy( mcv + i * ncheck * nmaxX2, mcv,  ncheck * nmaxX2 * sizeof(int) );
		memcpy( ref_mvc + i * nvar * nmaxX1, ref_mvc,  nvar * nmaxX1 * sizeof(int) );
	}

	cudaMalloc( (void**)&d_sumX1, nvar * sizeof(int) );		// const 64 K
	cudaMemcpy( d_sumX1, sumX1, nvar * sizeof(int), cudaMemcpyHostToDevice );

	cudaMalloc( (void**)&d_iind, nvar * nmaxX1 * sizeof(int) );		// const 1.2 M
	cudaMemcpy( d_iind, iind, nvar * nmaxX1 * sizeof(int), cudaMemcpyHostToDevice );

	cudaMalloc( (void**)&d_mcv, ncheck * nmaxX2 * sizeof(int) * N_FRAME );
	cudaMemcpy( d_mcv, mcv, ncheck * nmaxX2 * sizeof(int) * N_FRAME, cudaMemcpyHostToDevice );

	cudaMalloc( (void**)&d_input, nvar * sizeof(int) * N_FRAME );
	cudaMemcpy( d_input, input, nvar * sizeof(int) * N_FRAME, cudaMemcpyHostToDevice );
	
	cudaMalloc( (void**)&d_output, nvar * sizeof(char) * N_FRAME );
	cudaMemset( d_output, 0, nvar * sizeof(char) * N_FRAME );

	cudaMalloc( (void**)&d_mvc, nvar * nmaxX1 * sizeof(int) * N_FRAME );
	cudaMemset( d_mvc, 0, nvar * nmaxX1 * sizeof(int) * N_FRAME );

}

driverUpdataVar::~driverUpdataVar()
{
	// host
	free(sumX1);
	free(iind);	
	free(mvc);		free(mcv);
	free(input);	free(output);

	free(ref_mvc);	free(ref_output);

	// device
	cudaFree( d_sumX1 );
	cudaFree( d_iind );	
	cudaFree( d_mvc );		cudaFree( d_mcv );
	cudaFree( d_input );	cudaFree( d_output );
}