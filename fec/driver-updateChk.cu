
#include "driver-updateChk.cuh"

#include <cuda_runtime.h>

#include <fstream>
#include <iostream>
using namespace std;


#define		TABLE_SIZE_DINT2	300
#define		MAX_LOCAL_CACHE		30
#define		SIZE_BLOCK			256
#define		SIZE_BLOCK_2D_X		32

#define		USE_BLOCK_2D		0
#define		N_FRAME				10	// time scales as long as data length scales


__device__
int  logexp_device(const int x,
	const short int Dint1, const short int Dint2, const short int Dint3	//! Decoder (lookup-table) parameters
	, int* s_logexp_table )		
{
	int ind = x >> Dint3;
	if (ind >= Dint2) // outside table
		return 0;

	// Without interpolation
	return s_logexp_table[ind];
}

__device__
int Boxplus(const int a, const int b,
	const short int Dint1, const short int Dint2, const short int Dint3,	//! Decoder (lookup-table) parameters
	const int QLLR_MAX, int* s_logexp_table )		//! The lookup tables for the decoder
{
	//return a+b;
	int a_abs = (a > 0 ? a : -a);
	int b_abs = (b > 0 ? b : -b);
	int minabs = (a_abs > b_abs ? b_abs : a_abs);
	int term1 = (a > 0 ? (b > 0 ? minabs : -minabs)
		: (b > 0 ? -minabs : minabs));

	if (Dint2 == 0) {  // logmax approximation - avoid looking into empty table
		// Don't abort when overflowing, just saturate the QLLR
		if (term1 > QLLR_MAX) {
			return QLLR_MAX;
		}
		if (term1 < -QLLR_MAX) {
			return -QLLR_MAX;
		}
		return term1;
	}

	int apb = a + b;
	int term2 = logexp_device((apb > 0 ? apb : -apb), Dint1, Dint2, Dint3, s_logexp_table );
	int amb = a - b;
	int term3 = logexp_device((amb > 0 ? amb : -amb), Dint1, Dint2, Dint3, s_logexp_table );
	int result = term1 + term2 - term3;

	// Don't abort when overflowing, just saturate the QLLR
	if (result > QLLR_MAX) {
		return QLLR_MAX;
	}
	if (result < -QLLR_MAX) {
		return -QLLR_MAX;
	}
	return result;
}

__global__ 
void updateCheckNodeOpti_kernel( const int ncheck, const int nvar, 
	const int* sumX2, const int* n_mvc, const int* jind, int* logexp_table, 
	const short int Dint1, const short int Dint2, const short int Dint3, 
	const int QLLR_MAX,
	int* n_mcv,
	int nmaxX1, int nmaxX2, int nFrame )
{	//	mvc const(input)-> mcv (output)
	int j = blockIdx.x * blockDim.x + threadIdx.x;

	if( j>= ncheck )
		return;

	__shared__ int s_logexp_table[TABLE_SIZE_DINT2];

	int SIZE_BLOCK_TRY = 137;	//	uppermost 137, > 138 fail on Tesla 2050 
	if( threadIdx.x < SIZE_BLOCK_TRY )	{
	for( int ii=0; threadIdx.x + ii * SIZE_BLOCK_TRY < TABLE_SIZE_DINT2; ii++ )
		s_logexp_table[threadIdx.x + ii * SIZE_BLOCK_TRY] = logexp_table[threadIdx.x + ii * SIZE_BLOCK_TRY];
	}
	__syncthreads();

	int ml[MAX_LOCAL_CACHE];//int* ml	= d_ml	+ j * max_cnd;
	int mr[MAX_LOCAL_CACHE];//int* mr	= d_mr	+ j * max_cnd;
	int m[MAX_LOCAL_CACHE];

	
	for( int frame = 0; frame < nFrame; frame ++ )	{

	const int	*mvc	= n_mvc + frame * nvar * nmaxX1;
	int			*mcv	= n_mcv + frame * ncheck * nmaxX2;

	switch( sumX2[j] )
	{
	case 6:		{	
		int j0 = j;
		int m0 = mvc[jind[j0]];
		int j1 = j0 + ncheck;
		int m1 = mvc[jind[j1]];
		int j2 = j1 + ncheck;
		int m2 = mvc[jind[j2]];
		int j3 = j2 + ncheck;
		int m3 = mvc[jind[j3]];
		int j4 = j3 + ncheck;
		int m4 = mvc[jind[j4]];
		int j5 = j4 + ncheck;
		int m5 = mvc[jind[j5]];
		int m01 = Boxplus(m0, m1, Dint1, Dint2, Dint3, QLLR_MAX, s_logexp_table);
		int m23 = Boxplus(m2, m3, Dint1, Dint2, Dint3, QLLR_MAX, s_logexp_table);
		int m45 = Boxplus(m4, m5, Dint1, Dint2, Dint3, QLLR_MAX, s_logexp_table);
		int m03 = Boxplus(m01, m23, Dint1, Dint2, Dint3, QLLR_MAX, s_logexp_table);
		int m25 = Boxplus(m23, m45, Dint1, Dint2, Dint3, QLLR_MAX, s_logexp_table);
		int m0145 = Boxplus(m01, m45, Dint1, Dint2, Dint3, QLLR_MAX, s_logexp_table);
		mcv[j0] = Boxplus(m1, m25, Dint1, Dint2, Dint3, QLLR_MAX, s_logexp_table);
		mcv[j1] = Boxplus(m0, m25, Dint1, Dint2, Dint3, QLLR_MAX, s_logexp_table);
		mcv[j2] = Boxplus(m0145, m3, Dint1, Dint2, Dint3, QLLR_MAX, s_logexp_table);
		mcv[j3] = Boxplus(m0145, m2, Dint1, Dint2, Dint3, QLLR_MAX, s_logexp_table);
		mcv[j4] = Boxplus(m03, m5, Dint1, Dint2, Dint3, QLLR_MAX, s_logexp_table);
		mcv[j5] = Boxplus(m03, m4, Dint1, Dint2, Dint3, QLLR_MAX, s_logexp_table);
	}// case 6
	break;

	default:		{
	//if( j== ncheck )
	{
		for(int i = 0; i < sumX2[j]; i++ ) 
			m[i] = mvc[ jind[j+i*ncheck] ];
	}

	int nodes = sumX2[j];

	nodes--;

	// compute partial sums from the left and from the right
	//if( j== ncheck )
	{
		ml[0] = m[0];
		mr[0] = m[nodes];
		for(int i = 1; i < nodes; i++ ) {
			ml[i] = Boxplus( ml[i-1], m[i], Dint1, Dint2, Dint3, QLLR_MAX, s_logexp_table );
			mr[i] = Boxplus( mr[i-1], m[nodes-i], Dint1, Dint2, Dint3, QLLR_MAX, s_logexp_table );
		}
	}
	// merge partial sums
	//if( j== ncheck )
	{	
		mcv[j] = mr[nodes-1];
		mcv[j+nodes*ncheck] = ml[nodes-1];
		for(int i = 1; i < nodes; i++ )
			mcv[j+i*ncheck] = Boxplus( ml[i-1], mr[nodes-1-i], Dint1, Dint2, Dint3, QLLR_MAX, s_logexp_table );
	}
	}// default
	}//switch

	}// nFrame
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


bool driverUpdataChk::launch()
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

	updateCheckNodeOpti_kernel<<< grid, block >>>( ncheck, nvar, 
		d_sumX2, d_mvc, d_jind, d_logexp_table,
		Dint1, Dint2, Dint3, QLLR_MAX,
		d_mcv, nmaxX1, nmaxX2, N_FRAME );

#endif

	cudaError_t	status = cudaGetLastError();
	return status == cudaSuccess ;
}

bool driverUpdataChk::verify()
{
	cudaMemcpy( mcv, d_mcv, ncheck * nmaxX2 * sizeof(int) * N_FRAME, cudaMemcpyDeviceToHost );

	// mcv
	int i = 0;
	for ( ; i < ncheck * nmaxX2 * N_FRAME; i++ )
	{
		if ( ref_mcv[i] != mcv[i] )
			break;
	}

	if ( i < ncheck * nmaxX2 * N_FRAME )
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

driverUpdataChk::driverUpdataChk()
{
	Dint1 = 12;	Dint2 = 300;	Dint3 = 7;	//! Decoder (lookup-table) parameters
	QLLR_MAX = (1<<31 -1)>>4;//(std::numeric_limits<int>::max() >> 4);

	readFile( nvar, ncheck, nmaxX1, nmaxX2, "../data/ldpcSize.txt" );

	sumX2 = (int*)malloc(ncheck * sizeof(int));
	jind = (int*)malloc(ncheck * nmaxX2 * sizeof(int));
	mvc = (int*)malloc(nvar * nmaxX1 * sizeof(int) * N_FRAME);
	mcv = (int*)malloc(ncheck * nmaxX2 * sizeof(int) * N_FRAME);
	logexp_table = (int*)malloc(Dint2 * sizeof(int) );
	ref_mcv = (int*)malloc(ncheck * nmaxX2 * sizeof(int) * N_FRAME);

	ifstream  testfile;
	testfile.open( "../data/sumX2.txt" );
	if ( testfile == NULL )
	{
		cout << "Missing ldpc code parameter file - \"sumX2.txt\" in data path!" << endl ;
		return ;
	}
	else
	{
		cout << "Success to load ldpc code parameter file !" << endl ;
	}
	testfile.close();

	readArray( sumX2, ncheck, "../data/sumX2.txt" );

	readArray( jind, ncheck * nmaxX2, "../data/jind.txt" );

	readArray( ref_mcv, ncheck * nmaxX2, "../data/mcv.txt" );	

	readArray( mvc, nvar * nmaxX1, "../data/mvcInit.txt" );		

	readArray( logexp_table, Dint2, "../data/logexp.txt" );

	for( int i = 0; i < N_FRAME; i ++ )
	{
		memcpy( ref_mcv + i * ncheck * nmaxX2, ref_mcv,  ncheck * nmaxX2 * sizeof(int) );
		memcpy( mvc + i * nvar * nmaxX1, mvc,  nvar * nmaxX1 * sizeof(int) );
	}

	cudaMalloc( (void**)&d_sumX2, ncheck * sizeof(int) );	// const 32 K
	cudaMemcpy( d_sumX2, sumX2, ncheck * sizeof(int), cudaMemcpyHostToDevice );

	cudaMalloc( (void**)&d_jind, ncheck * nmaxX2 * sizeof(int) );	// const 300 K
	cudaMemcpy( d_jind, jind, ncheck * nmaxX2 * sizeof(int), cudaMemcpyHostToDevice );

	cudaMalloc( (void**)&d_mcv, ncheck * nmaxX2 * sizeof(int) * N_FRAME );
	cudaMemset( d_mcv, 0, ncheck * nmaxX2 * sizeof(int) * N_FRAME );

	cudaMalloc( (void**)&d_mvc, nvar * nmaxX1 * sizeof(int) * N_FRAME );
	cudaMemcpy( d_mvc, mvc, nvar * nmaxX1 * sizeof(int) * N_FRAME, cudaMemcpyHostToDevice );

	cudaMalloc( (void**)&d_logexp_table, Dint2 * sizeof(int) );		// const 1.2 K
	cudaMemcpy( d_logexp_table, logexp_table, Dint2 * sizeof(int), cudaMemcpyHostToDevice );

}

driverUpdataChk::~driverUpdataChk()
{
	// host
	free(sumX2);
	free(jind);
	free(mvc);		free(mcv);

	free(ref_mcv);
	free(logexp_table);

	// device
	cudaFree( d_sumX2 );
	cudaFree( d_jind );
	cudaFree( d_mvc );		cudaFree( d_mcv );
	cudaFree( d_logexp_table );
}