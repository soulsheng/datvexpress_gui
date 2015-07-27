
#include "ldpc_bp_decode.cuh"
#include "ldpc_bp_decode_kernel.cuh"
//#include "driverUtility.h"
#include "itppUtility.h"
#include "dvbUtility.h"

#include <cuda_runtime.h>
#include <thrust/reduce.h>
#include <thrust/device_vector.h>
#include <iostream>
using namespace std;
#if USE_TEXTURE_ADDRESS
	cudaArray* arr_mcv;
	cudaArray* arr_mvc;
	cudaChannelFormatDesc channelDesc;
#endif
#include "helper_timer.h"
#define		TIME_STEP		6	
#define		ENABLE_TIME_TEST		0	

bool ldpc_gpu::syndrome_check_gpu( int nMulti ) 
{// 0.07 ms/1f, 
	dim3 block( SIZE_BLOCK );
	dim3 grid( (ncheck + block.x - 1) / block.x );

	syndrome_check_kernel<<< grid, block >>>( d_LLRout, m_ldpcCurrent->d_sumX2, ncheck, m_ldpcCurrent->d_V, d_synd, nvar, nMulti );

	int h_synd=0;
	cudaMemcpy( &h_synd, d_synd, sizeof(int), cudaMemcpyDeviceToHost );

	return h_synd == 0;   // codeword is valid
}

void ldpc_gpu::updateVariableNode_gpu() 
{
	dim3 block( SIZE_BLOCK );
	dim3 grid( (nvar + block.x - 1) / block.x );

	updateVariableNode_kernel<<< grid, block >>>( nvar, ncheck, m_ldpcCurrent->d_sumX1, m_ldpcCurrent->d_mcv, m_ldpcCurrent->d_iind, d_LLRin, d_LLRout, m_ldpcCurrent->d_mvc );
}

void ldpc_gpu::updateCheckNode_gpu()
{
	dim3 block( SIZE_BLOCK );
	dim3 grid( (ncheck + block.x - 1) / block.x );

	updateCheckNode_kernel<<< grid, block >>>(ncheck, nvar, 
		m_ldpcCurrent->d_sumX2, m_ldpcCurrent->d_mvc, m_ldpcCurrent->d_jind, m_ldpcCurrent->d_logexp_table, 
		m_ldpcCurrent->Dint1, m_ldpcCurrent->Dint2, m_ldpcCurrent->Dint3,
		QLLR_MAX, m_ldpcCurrent->d_mcv );	// Shared not faster
}

void ldpc_gpu::initializeMVC_gpu( )
{
	dim3 block( 256 );
	dim3 grid( (nvar + block.x - 1) / block.x );

	initializeMVC_kernel<<< grid, block >>>( nvar, m_ldpcCurrent->d_sumX1, d_LLRin, m_ldpcCurrent->d_mvc, m_ldpcCurrent->nmaxX1, 1 );
}

int ldpc_gpu::bp_decode(int *LLRin, int *LLRout,
	bool psc /*= true*/,			//!< check syndrom after each iteration
	int max_iters /*= 50*/ )		//!< Maximum number of iterations
{
	cudaMemcpy( d_LLRin, LLRin, nvar * sizeof(int), cudaMemcpyHostToDevice );

  // initial step
	initializeMVC_gpu();

  bool is_valid_codeword = false;
  int iter = 0;
  do {
    iter++;
    //if (nvar >= 100000) { it_info_no_endl_debug("."); }
    // --------- Step 1: check to variable nodes ----------
	updateCheckNode_gpu();

#if USE_TEXTURE_ADDRESS
    // update the array to the texture
    cudaMemcpyToArray(arr_mcv, 0, 0, d_mcv, ncheck * nmaxX2 * sizeof(int), cudaMemcpyDeviceToDevice);
#endif

    // step 2: variable to check nodes
	updateVariableNode_gpu();

#if USE_TEXTURE_ADDRESS
    // update the array to the texture
    cudaMemcpyToArray(arr_mvc, 0, 0, d_mvc, nvar * nmaxX1 * sizeof(int), cudaMemcpyDeviceToDevice);
#endif

#if	USE_TABLE_CODE
	updateConstantMemoryLLRByte( d_LLRout );
#endif

	if (psc && syndrome_check_gpu()) {
	  is_valid_codeword = true;
      break;
    }
  }
  while (iter < max_iters);

  cudaMemcpy( LLRout, d_LLRout, nvar * sizeof(int), cudaMemcpyDeviceToHost );


  return (is_valid_codeword ? iter : -iter);
}

int ldpc_gpu::bp_decode_once(char *LLRout, int code_rate, 
	int *LLRin/*= NULL*/, 
	bool psc /*= true*/,			//!< check syndrom after each iteration
	int max_iters /*= 50*/ )		//!< Maximum number of iterations
{
	m_ldpcCurrent = m_ldpcDataPool.findLDPC_DATA( code_rate );
	nvar = m_ldpcCurrent->nvar;
	ncheck = m_ldpcCurrent->ncheck;

	if( LLRin )
	cudaMemcpy( d_LLRin, LLRin, nvar * sizeof(int), cudaMemcpyHostToDevice );

 	dim3 block( SIZE_BLOCK );
	dim3 grid( (nvar + block.x - 1) / block.x );

	// initial step
	initializeMVC_kernel<<< grid, block >>>( nvar, m_ldpcCurrent->d_sumX1, d_LLRin, m_ldpcCurrent->d_mvc, m_ldpcCurrent->nmaxX1, 1  );

#if WRITE_FILE_FOR_DRIVER
	static bool bRunOnce1 = false;
	if( !bRunOnce1 ){
		cudaMemcpy( h_mvc, m_ldpcCurrent->d_mvc, nvar * m_ldpcCurrent->nmaxX1 * sizeof(int), cudaMemcpyDeviceToHost );
		writeArray( h_mvc, nvar * m_ldpcCurrent->nmaxX1, "../data/mvcInit.txt" );
	
		cudaMemcpy( h_LLRin, d_LLRin, nvar * sizeof(int), cudaMemcpyDeviceToHost );
		writeArray( h_LLRin, nvar, "../data/input.txt" );
		bRunOnce1 = true;
	}
#endif

	int not_valid_codeword = true;
	int iter = 1;
	for( ; iter < max_iters && not_valid_codeword; iter ++ )
	{
		// --------- Step 1: check to variable nodes ----------
		updateCheckNodeOpti_kernel<<< grid, block >>>(ncheck, nvar, 
			m_ldpcCurrent->d_sumX2, m_ldpcCurrent->d_mvc,
			m_ldpcCurrent->d_jind, m_ldpcCurrent->d_logexp_table, 
			m_ldpcCurrent->Dint1, m_ldpcCurrent->Dint2, m_ldpcCurrent->Dint3,QLLR_MAX, 
			m_ldpcCurrent->nmaxX1, m_ldpcCurrent->nmaxX2, 1, 
			m_ldpcCurrent->d_mcv );	// Shared not faster

				
#if WRITE_FILE_FOR_DRIVER
		static bool bRunOnce1 = false;
		if( iter == 1 && !bRunOnce1 ){
			cudaMemcpy( h_mcv, m_ldpcCurrent->d_mcv, ncheck * m_ldpcCurrent->nmaxX2 * sizeof(int), cudaMemcpyDeviceToHost );

			writeArray( h_mcv, ncheck * m_ldpcCurrent->nmaxX2, "../data/mcv.txt" );

			bRunOnce1 = true;
		}
#endif

		// --------- Step 2: variable to check nodes ----------
#if USE_BLOCK_2D
	
	dim3 block( SIZE_BLOCK_2D_X, MAX_VAR_NODE );
	dim3 grid;
	grid.x = (nvar * MAX_VAR_NODE + SIZE_BLOCK_2D_X * MAX_VAR_NODE - 1) 
				/ (SIZE_BLOCK_2D_X * MAX_VAR_NODE) ;

	updateVariableNodeOpti2D_kernel<<< grid, block >>>( nvar, ncheck, 
		d_sumX1, d_mcv, d_iind, d_LLRin, 
		d_LLRout, d_mvc );
#else
		updateVariableNodeOpti_kernel<<< grid, block >>>( nvar, ncheck, 
			m_ldpcCurrent->d_sumX1, m_ldpcCurrent->d_mcv, m_ldpcCurrent->d_iind, d_LLRin, 
			m_ldpcCurrent->nmaxX1, m_ldpcCurrent->nmaxX2, 1, 
			d_LLRout, m_ldpcCurrent->d_mvc );
#endif

		// --------- Step 3: check syndrome 奇偶校验 ----------
#if 0
		syndrome_check_kernel<<< grid, block >>>( d_LLRout, d_sumX2, ncheck, d_V, 
			d_synd );

		cudaMemcpy( &not_valid_codeword, d_synd, sizeof(int), cudaMemcpyDeviceToHost );
#else
		cudaMemcpy( LLRout, d_LLRout, nvar * sizeof(char), cudaMemcpyDeviceToHost );
		
#if WRITE_FILE_FOR_DRIVER
		static bool bRunOnce2 = false;
		if( iter == 1 && !bRunOnce2 ){
			cudaMemcpy( h_mvc, m_ldpcCurrent->d_mvc, nvar * m_ldpcCurrent->nmaxX1 * sizeof(int), cudaMemcpyDeviceToHost );

			writeArray( LLRout, nvar, "../data/output.txt" );
			writeArray( h_mvc, nvar * m_ldpcCurrent->nmaxX1, "../data/mvc.txt" );		

			bRunOnce2 = true;
		}
#endif

		if (psc && check_parity_cpu(LLRout)) {
			 not_valid_codeword = false;
			break;
		}
#endif
	}
  
  //cudaMemcpy( LLRout, d_LLRout, nvar * sizeof(char), cudaMemcpyDeviceToHost );


  return (!not_valid_codeword ? iter : -iter);
}

bool ldpc_gpu::check_parity_cpu(char *LLRN, int nFrame) 
{
	// Please note the IT++ convention that a sure zero corresponds to
	// LLR=+infinity
	int i, j, synd, vi;

	for( int frame = 0; frame < nFrame; frame ++ ) {
	char *LLR = LLRN + nvar * frame;

	for (j = 0; j < ncheck; j++) {
		synd = 0;
		int vind = j; // tracks j+i*ncheck
		for (i = 0; i < m_ldpcCurrent->h_sumX2[j]; i++) {
			vi = m_ldpcCurrent->h_V[vind];
			if (LLR[vi]) {
				synd++;
			}
			vind += ncheck;
		}
		if ((synd&1) == 1) {
			return false;  // codeword is invalid
		}
	}
	}
	return true;   // codeword is valid
}

bool ldpc_gpu::initialize( LDPC_CodeFactory* pcodes, scmplx* psymbols, int nMultiMax )
{
	m_nMultiMax = nMultiMax;

	m_ldpcDataPool.initialize( pcodes, nMultiMax );
	LDPC_DATA_GPU* m_ldpcCurrent = m_ldpcDataPool.findLDPC_DATA(0);
	nvar = m_ldpcCurrent->nvar;
	ncheck = m_ldpcCurrent->ncheck;

	//max_cnd = 200;
	QLLR_MAX = (1<<31 -1)>>4;//(std::numeric_limits<int>::max() >> 4);

	cudaMalloc( (void**)&d_LLRin, m_nMultiMax * nvar * sizeof(int) );
	cudaMalloc( (void**)&d_LLRout, m_nMultiMax * nvar * sizeof(char) );
	cudaMemset( d_LLRout, 1, m_nMultiMax * nvar * sizeof(char) );

	cudaMalloc( (void**)&d_synd, nMultiMax * sizeof(int) );
	cudaMemset( d_synd, 0, nMultiMax * sizeof(int) );
	
	cudaMalloc( (void**)&d_pSymbolsTemplate, M_SYMBOL_SIZE_MAX * M_CONST_NUMBER * sizeof(scmplx) );
	cudaMemcpy( d_pSymbolsTemplate, psymbols, M_SYMBOL_SIZE_MAX * M_CONST_NUMBER * sizeof(scmplx), cudaMemcpyHostToDevice );
	m_pSymbolsTemplate = psymbols;

	cudaMalloc( (void**)&d_pSymbolsIn, m_nMultiMax * FRAME_SIZE_NORMAL * sizeof(scmplx) );

	cudaMalloc( (void**)&d_pDist2, m_nMultiMax * M_SYMBOL_SIZE_MAX * FRAME_SIZE_NORMAL * sizeof(float) );

	cudaMalloc( (void**)&d_pSoftBitCache, m_nMultiMax * FRAME_SIZE_NORMAL * sizeof(int) );

	m_pDist2 = new float[m_nMultiMax*FRAME_SIZE_NORMAL*M_SYMBOL_SIZE_MAX];

	initConstantMemoryLogExp(m_ldpcCurrent->getCode()->llrcalc.logexp_table._data());

#if USE_TEXTURE_ADDRESS
	// cuda texture ------------------------------------------------------------------------------------------
	channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindSigned);
    cudaError_t err = cudaMallocArray(&arr_mcv, &channelDesc, ncheck, nmaxX2);
    cudaMemcpyToArray(arr_mcv, 0, 0, d_mcv, ncheck * nmaxX2 * sizeof(int), cudaMemcpyDeviceToDevice);

	texMCV.addressMode[0] = cudaAddressModeClamp;
	texMCV.addressMode[1] = cudaAddressModeClamp;
    texMCV.filterMode = cudaFilterModePoint;
    texMCV.normalized = false;

	cudaBindTextureToArray(texMCV, arr_mcv, channelDesc);

	cudaMallocArray(&arr_mvc, &channelDesc, nvar, nmaxX1);
    cudaMemcpyToArray(arr_mvc, 0, 0, d_mvc, nvar * nmaxX1 * sizeof(int), cudaMemcpyDeviceToDevice);
	cudaBindTextureToArray(texMVC, arr_mvc, channelDesc);

#endif

	h_mvc = (int*)malloc(nvar * MAX_LOCAL_CACHE * sizeof(int));
	h_mcv = (int*)malloc(ncheck * MAX_LOCAL_CACHE * sizeof(int));
	h_LLRin= (int*)malloc(nvar * sizeof(int));
	h_pSoftBit = (int*)malloc(m_nMultiMax * FRAME_SIZE_NORMAL* sizeof(int));

	return true;
}


void ldpc_gpu::release()
{
	if( NULL == d_LLRin )
		return;

	cudaFree( d_LLRin );	cudaFree( d_LLRout );
	d_LLRin = NULL;			d_LLRout = NULL;

	cudaFree( d_synd );
	d_synd = NULL;

	cudaFree( d_pSymbolsTemplate );
	d_pSymbolsTemplate = NULL;

	cudaFree( d_pDist2 );	d_pDist2 = NULL;

	free( h_mvc );	free( h_mcv );	free( h_LLRin );
	h_mvc = NULL;	h_mcv = NULL;	h_LLRin = NULL;
	free( m_pDist2 );
	m_pDist2 = NULL;

	return ;
}

ldpc_gpu::~ldpc_gpu()
{
	release();
}

int ldpc_gpu::bp_decode_once( itpp::vec& softbits, char *LLRout, int code_rate )
{
	m_ldpcCurrent = m_ldpcDataPool.findLDPC_DATA( code_rate );
	nvar = m_ldpcCurrent->nvar;
	ncheck = m_ldpcCurrent->ncheck;
	itpp::QLLRvec llrIn = m_ldpcCurrent->getCode()->get_llrcalc().to_qllr(softbits);

	return bp_decode_once( LLRout, code_rate, llrIn._data() );	
}

int ldpc_gpu::bp_decode_once( double* softbits, char *LLRout, int code_rate )
{
	m_ldpcCurrent = m_ldpcDataPool.findLDPC_DATA( code_rate );
	nvar = m_ldpcCurrent->nvar;
	ncheck = m_ldpcCurrent->ncheck;
	itpp::vec  softVec( nvar );
	convertBufferToVec( softbits, softVec );
	return bp_decode_once( softVec, LLRout, code_rate );
}

float ldpc_gpu::distance( const scmplx& cL, const scmplx& cR )
{
	float dist2 = 0;
	dist2 = (float)(cL.im - cR.im) * (cL.im - cR.im) + (float)(cL.re - cR.re) * (cL.re - cR.re);
	return dist2;
}

int ldpc_gpu::decode_soft( scmplx* sym, double N0, int nPayloadSymbols, int M, int k,
	int *pFrame, int code_rate,
	double* p_soft_bits, double* p_soft_bits_cache,
	char* p_bitLDPC, int nMulti /*= 1*/ )
{
	m_ldpcCurrent = m_ldpcDataPool.findLDPC_DATA( code_rate );
	nvar = m_ldpcCurrent->nvar;
	ncheck = m_ldpcCurrent->ncheck;

	for ( int j = 0;j<nMulti;j++ )
		cudaMemcpy( d_pSymbolsIn+j*nPayloadSymbols, 
			sym + j*FRAME_SIZE_NORMAL, 
			nPayloadSymbols * sizeof(scmplx), 
			cudaMemcpyHostToDevice );// 0.02 ms/1f, 0.28 ms/3f

	dim3 block( 1024/M );
	dim3 grid;
	grid.x = (nPayloadSymbols + block.x-1)/block.x;

	block.y = M;
	distance_kernel<<< grid, block >>>( d_pSymbolsIn, d_pSymbolsTemplate+(k-2)*M_SYMBOL_SIZE_MAX, M, d_pDist2, CP, nPayloadSymbols, nMulti );// 0.13 ms/1f, 0.28 ms/3f

	block.y = k;
	soft_bit_kernel<<< grid, block >>>(d_pDist2, d_pSoftBitCache, k, M, N0, 
		m_ldpcCurrent->Dint1, QLLR_MAX, nPayloadSymbols, nMulti );// 0.1 ms/1f, 0.23 ms/3f

	
#if WRITE_FILE_FOR_DRIVER
	static bool bRunOnce1 = false;
	if( !bRunOnce1 ){
		std::vector<int*> params;
		params.push_back(&k);
		params.push_back(&M);
		int Dint1 = m_ldpcCurrent->Dint1;
		params.push_back(&Dint1);
		params.push_back(&QLLR_MAX);
		params.push_back(&nPayloadSymbols);
		params.push_back(&nMulti);
		writeFile( params, "../data/softBitSize.txt" );

		cudaMemcpy( m_pDist2, d_pDist2, nMulti * M_SYMBOL_SIZE_MAX * FRAME_SIZE_NORMAL * sizeof(float), cudaMemcpyDeviceToHost );
		writeArray( m_pDist2, nMulti * M_SYMBOL_SIZE_MAX * FRAME_SIZE_NORMAL, "../data/pDist2.txt" );
	
		cudaMemcpy( h_pSoftBit, d_pSoftBitCache, nMulti * FRAME_SIZE_NORMAL * sizeof(int), cudaMemcpyDeviceToHost );
		writeArray( h_pSoftBit, nMulti * FRAME_SIZE_NORMAL, "../data/pSoftBit.txt" );
		bRunOnce1 = true;
	}
#endif

	// step	2:	de-interleave
#if 1
	reorder_kernel<<< grid, block >>>(d_LLRin, d_pSoftBitCache, k, nPayloadSymbols, nMulti );// 0.02 ms/1f, 0.03 ms/3f
	//cudaMemcpy( p_soft_bits, d_LLRin, FRAME_SIZE_NORMAL*sizeof(int), cudaMemcpyDeviceToHost );
#else
	cudaMemcpy( p_soft_bits_cache, d_pSoftBitCache, FRAME_SIZE_NORMAL*sizeof(int), cudaMemcpyDeviceToHost );
	for( int i = 0; i < nPayloadSymbols; i++ )
		for (int j=0;j<k;j++)
			p_soft_bits[j*nPayloadSymbols+i] = p_soft_bits_cache[i*k+j];	
#endif

	// step	3:	ldpc decode
	bp_decode_once( p_bitLDPC, code_rate, nMulti );// 1.8 ms/1f, 5.0 ms/3f

	// step	4:	cast type, char -> int
	for( int frame = 0; frame < nMulti; frame ++ ) {
	for( int i = 0; i < nvar; i++ )
		pFrame[i+frame*FRAME_SIZE_NORMAL] = p_bitLDPC[i+frame*nvar];

	}

	cudaError_t ce = cudaGetLastError();
	if( cudaSuccess != ce )
		cout << "bp_decode_once cudaGetLastError = " << ce << endl;

	return 0;
}

void	ldpc_gpu::updateSymbolsTemplate( scmplx* psymbols )
{
	cudaMemcpy( d_pSymbolsTemplate, psymbols, M_SYMBOL_SIZE_MAX * M_CONST_NUMBER * sizeof(scmplx), cudaMemcpyHostToDevice );
}

int ldpc_gpu::bp_decode_once(char *LLRout, int code_rate, int nMulti, 
	int *LLRin/*= NULL*/, 
	bool psc /*= true*/,			//!< check syndrom after each iteration
	int max_iters /*= 50*/ )		//!< Maximum number of iterations
{
	StopWatchInterface	*timerStep;

	sdkCreateTimer( &timerStep );

	m_ldpcCurrent = m_ldpcDataPool.findLDPC_DATA( code_rate );
	nvar = m_ldpcCurrent->nvar;
	ncheck = m_ldpcCurrent->ncheck;

	if( LLRin )
	cudaMemcpy( d_LLRin, LLRin, nvar * sizeof(int), cudaMemcpyHostToDevice );

 	dim3 block( SIZE_BLOCK );
	dim3 grid( (nvar + block.x - 1) / block.x );

	// initial step 0.04ms
	initializeMVC_kernel<<< grid, block >>>( nvar, m_ldpcCurrent->d_sumX1, d_LLRin, m_ldpcCurrent->d_mvc, m_ldpcCurrent->nmaxX1, nMulti );

	int not_valid_codeword = true;
	int iter = 1;
	for( ; iter < max_iters && not_valid_codeword; iter ++ )
	{
		// --------- Step 1: check to variable nodes ----------	0.31 ms
		updateCheckNodeOpti_kernel<<< grid, block >>>(ncheck, nvar, 
			m_ldpcCurrent->d_sumX2, m_ldpcCurrent->d_mvc,
			m_ldpcCurrent->d_jind, m_ldpcCurrent->d_logexp_table, 
			m_ldpcCurrent->Dint1, m_ldpcCurrent->Dint2, m_ldpcCurrent->Dint3,QLLR_MAX, 
			m_ldpcCurrent->nmaxX1, m_ldpcCurrent->nmaxX2, nMulti, 
			m_ldpcCurrent->d_mcv );	// Shared not faster


		// --------- Step 2: variable to check nodes ----------	0.23 ms
		updateVariableNodeOpti_kernel<<< grid, block >>>( nvar, ncheck, 
			m_ldpcCurrent->d_sumX1, m_ldpcCurrent->d_mcv, m_ldpcCurrent->d_iind, d_LLRin, 
			m_ldpcCurrent->nmaxX1, m_ldpcCurrent->nmaxX2, nMulti, 
			d_LLRout, m_ldpcCurrent->d_mvc );


		// --------- Step 3: check syndrome 奇偶校验 ---------- 0.11 ms	
#if ENABLE_TIME_TEST
		cudaDeviceSynchronize();
		sdkStartTimer( &timerStep );
#endif
		bool bVaid = syndrome_check_gpu( nMulti );// 0.17 ms 
		
#if ENABLE_TIME_TEST
		sdkStopTimer( &timerStep );
		cout  << "timerStepValue = "<< sdkGetTimerValue( &timerStep ) << " ms, " << endl;
#endif
	
		if (psc && bVaid ) {
			 not_valid_codeword = false;
			break;
		}

	}

	cudaMemcpy( LLRout, d_LLRout, nMulti * nvar * sizeof(char), cudaMemcpyDeviceToHost );

  return (!not_valid_codeword ? iter : -iter);
}
