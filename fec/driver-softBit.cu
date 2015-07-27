
#include "driver-softBit.cuh"
#include "modulatorDefinition.h"
#include <cuda_runtime.h>
#include "dvbUtility.h"

#include <fstream>
#include <iostream>
using namespace std;


#define		MAX_LOCAL_CACHE		20


#define		SIZE_BLOCK			256
#define		SIZE_BLOCK_2D_X		32

#define		USE_BLOCK_2D		0
#define		N_FRAME				10	// time scales as long as data length scales

__device__
double to_double(int l, int Dint1) 
{
  return static_cast<double>(l) / (1 << Dint1);
}

__device__
int to_qllr(double l, int Dint1, const int QLLR_MAX) 
{
  double QLLR_MAX_double = to_double(QLLR_MAX, Dint1);
  // Don't abort when overflow occurs, just saturate the QLLR
  if (l > QLLR_MAX_double) {
    return QLLR_MAX;
  }
  if (l < -QLLR_MAX_double) {
    return -QLLR_MAX;
  }
  return static_cast<int>(std::floor(0.5 + (1 << Dint1) * l));
}

__global__
void soft_bit_kernel(float *m_pDist2N, int *p_soft_bits_cacheN, int k, int M, float N0, 
	int Dint1, const int QLLR_MAX, int n, int nFrame )
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int i = threadIdx.y;

	if( index>=n )
		return;

	for( int frame = 0; frame < nFrame; frame ++ )	{

	float *m_pDist2 = m_pDist2N + frame*M*n;
	int *p_soft_bits_cache = p_soft_bits_cacheN + frame*k*n;

	double d0min, d1min, temp;

		//for (int i = 0; i < k; i++) 
		{
			d0min = d1min = 1<<20;

			for (int j = 0; j < M; j++) 
			{
				temp = m_pDist2[index+j*n];
				if ( j&(1<<(k-i-1)) )
				{
					if (temp < d1min) 
					{ 
						d1min = temp; 
					}
				}
				else
				{
					if (temp < d0min) 
					{ 
						d0min = temp; 
					}
				}
			}
			double l = (-d0min + d1min) / N0;
			p_soft_bits_cache[index*k + i] = to_qllr( l, Dint1, QLLR_MAX );

		}

	}
}

bool driverSoftBit::launch()
{

	dim3 block( 1024/M );
	dim3 grid;
	grid.x = (nPayloadSymbols + block.x-1)/block.x;

	block.y = k;
	soft_bit_kernel<<< grid, block >>>(d_pDist2, d_pSoftBitCache, k, M, N0, 
		Dint1, QLLR_MAX, nPayloadSymbols, nMulti );// 0.1 ms/1f, 0.23 ms/3f


	cudaError_t	status = cudaGetLastError();
	return status == cudaSuccess ;
}

bool driverSoftBit::verify()
{
	cudaMemcpy( m_pSoftBit, d_pSoftBitCache, nMulti * FRAME_SIZE_NORMAL * sizeof(int), cudaMemcpyDeviceToHost );

	// mvc
	int i = 0;
	for ( ; i < nMulti * FRAME_SIZE_NORMAL; i++ )
	{
		if ( ref_pSoftBit[i] != m_pSoftBit[i] )
			break;
	}

	if ( i < nMulti * FRAME_SIZE_NORMAL )
		return false;

	return true;
}

driverSoftBit::driverSoftBit( )
{
	//readFile( nvar, ncheck, nmaxX1, nmaxX2, "../data/ldpcSize.txt" );
	int m_nMultiMax = 10;
	//N0 = 0.01;
	N0 = pow(10.0, -EBNO / 10.0) / 0.75;

	//int k, M, Dint1, QLLR_MAX, n

	std::vector<int*> params;
	params.push_back(&k);
	params.push_back(&M);
	params.push_back(&Dint1);
	params.push_back(&QLLR_MAX);
	params.push_back(&nPayloadSymbols);
	params.push_back(&nMulti);
	readFile( params, "../data/softBitSize.txt" );

	m_pDist2 = new float[nMulti * FRAME_SIZE_NORMAL*M_SYMBOL_SIZE_MAX ];
	m_pSoftBit = new int[nMulti * FRAME_SIZE_NORMAL ];

	ref_pSoftBit = new int[nMulti * FRAME_SIZE_NORMAL ];
	
	ifstream  testfile;
	testfile.open( "../data/pDist2.txt" );
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

	readArray( m_pDist2, nMulti * FRAME_SIZE_NORMAL*M_SYMBOL_SIZE_MAX, "../data/pDist2.txt" );

	readArray( ref_pSoftBit, nMulti * FRAME_SIZE_NORMAL, "../data/pSoftBit.txt" );

	cudaMalloc( (void**)&d_pDist2, nMulti * M_SYMBOL_SIZE_MAX * FRAME_SIZE_NORMAL * sizeof(float) );
	cudaMemcpy( d_pDist2, m_pDist2, nMulti * M_SYMBOL_SIZE_MAX * FRAME_SIZE_NORMAL * sizeof(float), cudaMemcpyHostToDevice );

	cudaMalloc( (void**)&d_pSoftBitCache, nMulti * FRAME_SIZE_NORMAL * sizeof(int) );
	cudaMemset( d_pSoftBitCache, 0, nMulti * FRAME_SIZE_NORMAL * sizeof(int) );


}

driverSoftBit::~driverSoftBit()
{
	// host
	free(m_pDist2);			m_pDist2 = NULL;
	free(m_pSoftBit);		m_pSoftBit = NULL;
	free(ref_pSoftBit);		ref_pSoftBit = NULL;

	// device
	cudaFree( d_pDist2 );	d_pDist2 = NULL;
	cudaFree( d_pSoftBitCache );	d_pSoftBitCache = NULL;
}