
#pragma once

#define		TABLE_SIZE_DINT2	300
#define		MAX_LOCAL_CACHE		30
#define		USE_TABLE_CODE		0
#define		USE_TEXTURE_ADDRESS	0
#define		USE_SHARED_MLR		0
#define		USE_SHARED_MVC		0
#define		SIZE_BLOCK			256
#define		SIZE_BLOCK_2D_X		128
#define		USE_BLOCK_2D		0

#if USE_TEXTURE_ADDRESS
texture<int, 2, cudaReadModeElementType> texMCV;
texture<int, 2, cudaReadModeElementType> texMVC;
#endif

__device__ __constant__ int const_logexp_table[TABLE_SIZE_DINT2];

#if		USE_TABLE_CODE
__device__ __constant__ char const_llr_byte[VAR_SIZE_CODE];	// char

void updateConstantMemoryLLRByte(char *bLLR)
{
	cudaMemcpyToSymbol( const_llr_byte, bLLR, VAR_SIZE_CODE * sizeof(char), 0, cudaMemcpyDeviceToDevice );
}
#endif

void initConstantMemoryLogExp(int *logexp_table)
{
	cudaMemcpyToSymbol( const_logexp_table, logexp_table, TABLE_SIZE_DINT2 * sizeof(int), 0, cudaMemcpyHostToDevice );
}

__global__ 
void syndrome_check_kernel(const char *d_LLR,
	const int* d_sumX2, const int ncheck, 
	const int* d_V,
	int* d_synd) 
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;

	if( j>= ncheck )
		return;

	if( j == 0 )
		*d_synd = 0;

	__syncthreads();

	int i, vi;
	int synd = 0;
	int vind = j; // tracks j+i*ncheck
	for (i = 0; i < d_sumX2[j]; i++) {
		vi = d_V[vind];
#if	USE_TABLE_CODE
		if ( const_llr_byte[vi] ) {
#else
		if (d_LLR[vi]) {
#endif
			synd++;
		}
		vind += ncheck;
	}

	if( synd&1 )
		atomicAdd( d_synd, 1 );	// synd is even ?
}

__global__ 
void updateVariableNode_kernel( const int nvar, const int ncheck, const int* sumX1, const int* mcv, const int* iind, const int * LLRin, 
	char * LLRout, int* mvc ) 
{	//	mcv const(input)-> mvc (output)
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	
	if( i>= nvar )
		return;

	int mvc_temp = LLRin[i];
	for (int jp = 0; jp < sumX1[i]; jp++) {
#if USE_TEXTURE_ADDRESS
		int index = iind[i + jp*nvar];
		mvc_temp +=  tex2D(texMCV, index%ncheck, index/ncheck);
#else
		mvc_temp +=  mcv[ iind[i + jp*nvar] ];
#endif
	}
	LLRout[i] = mvc_temp<0;

	for (int jp = 0; jp < sumX1[i]; jp++)
		mvc[i + jp*nvar] = mvc_temp - mcv[ iind[i + jp*nvar] ];
	
}

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
void updateCheckNode_kernel( const int ncheck, const int nvar, 
	const int* sumX2, const int* mvc, const int* jind, int* logexp_table,  
	const short int Dint1, const short int Dint2, const short int Dint3, 
	const int QLLR_MAX,
	int* mcv )
{	//	mvc const(input)-> mcv (output)
	int j = blockIdx.x * blockDim.x + threadIdx.x;

	if( j>= ncheck )
		return;

	int ml[MAX_LOCAL_CACHE];//int* ml	= d_ml	+ j * max_cnd;
	int mr[MAX_LOCAL_CACHE];//int* mr	= d_mr	+ j * max_cnd;

	int nodes = sumX2[j];

	nodes--;

	// compute partial sums from the left and from the right
#if USE_TEXTURE_ADDRESS
	int indexL = jind[j];
	ml[0] = tex2D(texMVC, indexL%nvar, indexL/nvar);
	int indexR = jind[j+nodes*ncheck];
	mr[0] = tex2D(texMVC, indexR%nvar, indexR/nvar);
	for(int i = 1; i < nodes; i++ ) {
		indexL = jind[j+i*ncheck];
		int valueL = tex2D(texMVC, indexL%nvar, indexL/nvar);;
		indexR = jind[j+(nodes-i)*ncheck];
		int valueR = tex2D(texMVC, indexR%nvar, indexR/nvar);;

		ml[i] = Boxplus( ml[i-1], valueL, Dint1, Dint2, Dint3, QLLR_MAX );
		mr[i] = Boxplus( mr[i-1], valueR, Dint1, Dint2, Dint3, QLLR_MAX );
	}
#else
	ml[0] = mvc[jind[j]];
	mr[0] = mvc[jind[j+nodes*ncheck]];
	for(int i = 1; i < nodes; i++ ) {
		ml[i] = Boxplus( ml[i-1], mvc[jind[j+i*ncheck]], Dint1, Dint2, Dint3, QLLR_MAX, logexp_table );
		mr[i] = Boxplus( mr[i-1], mvc[jind[j+(nodes-i)*ncheck]], Dint1, Dint2, Dint3, QLLR_MAX, logexp_table );
	}
#endif
	// merge partial sums
	mcv[j] = mr[nodes-1];
	mcv[j+nodes*ncheck] = ml[nodes-1];
	for(int i = 1; i < nodes; i++ )
		mcv[j+i*ncheck] = Boxplus( ml[i-1], mr[nodes-1-i], Dint1, Dint2, Dint3, QLLR_MAX, logexp_table );

}

__global__ 
void updateCheckNodeShared_kernel( const int ncheck, const int nvar, 
	const int* sumX2, const int* mvc, const int* jind, int* logexp_table, 
	const short int Dint1, const short int Dint2, const short int Dint3, 
	const int QLLR_MAX,
	int* mcv )
{	//	mvc const(input)-> mcv (output)
	int j = blockIdx.x * blockDim.x + threadIdx.x;

	if( j>= ncheck )
		return;
#if USE_SHARED_MLR
	__shared__ int s_ml[MAX_CHECK_NODE*SIZE_BLOCK];
	int* ml	= s_ml	+ threadIdx.x * MAX_CHECK_NODE;
	__shared__ int s_mr[MAX_CHECK_NODE*SIZE_BLOCK];
	int* mr	= s_mr	+ threadIdx.x * MAX_CHECK_NODE;
#else
	int ml[MAX_LOCAL_CACHE];
	int mr[MAX_LOCAL_CACHE];
#endif

	int nodes = sumX2[j];

	nodes--;

	// compute partial sums from the left and from the right
#if USE_SHARED_MVC
	__shared__ int s_mvc[MAX_CHECK_NODE*SIZE_BLOCK];
	for(int i = 0; i < sumX2[j]; i++ ) {
		s_mvc[threadIdx.x+i*SIZE_BLOCK] = mvc[ jind[j+i*ncheck] ];
	}

	ml[0] = s_mvc[threadIdx.x];
	mr[0] = s_mvc[threadIdx.x+nodes*SIZE_BLOCK];
	for(int i = 1; i < nodes; i++ ) {
		ml[i] = Boxplus( ml[i-1], s_mvc[threadIdx.x+i*SIZE_BLOCK], Dint1, Dint2, Dint3, QLLR_MAX );
		mr[i] = Boxplus( mr[i-1], s_mvc[threadIdx.x+(nodes-i)*SIZE_BLOCK], Dint1, Dint2, Dint3, QLLR_MAX );
	}
#else
	ml[0] = mvc[jind[j]];
	mr[0] = mvc[jind[j+nodes*ncheck]];
	for(int i = 1; i < nodes; i++ ) {
		ml[i] = Boxplus( ml[i-1], mvc[jind[j+i*ncheck]], Dint1, Dint2, Dint3, QLLR_MAX, logexp_table );
		mr[i] = Boxplus( mr[i-1], mvc[jind[j+(nodes-i)*ncheck]], Dint1, Dint2, Dint3, QLLR_MAX, logexp_table );
	}
#endif

	// merge partial sums
	mcv[j] = mr[nodes-1];
	mcv[j+nodes*ncheck] = ml[nodes-1];
	for(int i = 1; i < nodes; i++ )
		mcv[j+i*ncheck] = Boxplus( ml[i-1], mr[nodes-1-i], Dint1, Dint2, Dint3, QLLR_MAX, logexp_table );

}

__global__ 
void initializeMVC_kernel(const int nvar, 
	const int* sumX1,
	const int* LLRinN,
	int* mvcN, int nmaxX1, int nFrame) 
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if( i>= nvar )
		return;

	for( int frame = 0; frame < nFrame; frame ++ ){
    const int* LLRin = LLRinN + frame * nvar;
	int* mvc = mvcN + frame * nvar * nmaxX1;

	int index = i;
    for (int j = 0; j < sumX1[i]; j++) {
      mvc[index] = LLRin[i];
      index += nvar;
    }

	}
}

__global__ 
void updateVariableNodeOpti_kernel( const int nvar, const int ncheck, const int* sumX1, const int* mcvN, const int* iind, const int * LLRinN, 
	const int nmaxX1, const int nmaxX2, const int nFrame, 
	char * LLRoutN, int* mvcN ) // not used, just for testing performance bound
{	//	mcv const(input)-> mvc (output)
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	
	if( i>= nvar )
		return;
	
	for( int frame = 0; frame < nFrame; frame ++ ){
    const int* LLRin = LLRinN + frame * nvar;
 	int* mvc = mvcN + frame * nvar * nmaxX1;
	const int* mcv = mcvN + frame * ncheck * nmaxX2;
    char* LLRout = LLRoutN + frame * nvar;

	int mvc_temp = LLRin[i];

	int m[MAX_LOCAL_CACHE];
	//if( threadIdx.x == nvar )		kernel updateVar read 3 misalligned global memory cost 80% time on tesla c2050
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
void updateCheckNodeOpti_kernel( const int ncheck, const int nvar, 
	const int* sumX2, const int* mvcN, const int* jind, int* logexp_table, 
	const short int Dint1, const short int Dint2, const short int Dint3, 
	const int QLLR_MAX,
	const int nmaxX1, const int nmaxX2, const int nFrame, 
	int* mcvN )
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

	
	for( int frame = 0; frame < nFrame; frame ++ ) {
 	const int* mvc = mvcN + frame * nvar * nmaxX1;
	int* mcv = mcvN + frame * ncheck * nmaxX2;

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
	}//for
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

__global__
void distance_kernel(scmplx *sym, scmplx *symTemplate, int M, float *dist2N, 
	int scale, int n, int nFrame )
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = threadIdx.y;

	if( i>=n )
		return;

	for( int frame = 0; frame < nFrame; frame ++ )	{

	float *dist2 = dist2N + frame*M*n; 
	scmplx symIn = sym[i + frame * n];

	//for (int j = 0; j < M; j++) 
	{
		scmplx symTp = symTemplate[j];

		double dist2Val = (symIn.re*1.0f - symTp.re) * (symIn.re - symTp.re) 
			+ (symIn.im*1.0f - symTp.im) * (symIn.im - symTp.im);

		dist2Val /= scale*scale;

		dist2[i*M+j] = dist2Val;

	}

	}
}

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

#if 1
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
				temp = m_pDist2[index*M+j];
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
#endif
__global__
void reorder_kernel(int *p_soft_bitsN, int *p_soft_bits_cacheN, int k, int nPayloadSymbols, int nFrame )
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int j = threadIdx.y;
	//for (int j = 0;j < k; j++) 
	for( int frame = 0; frame < nFrame; frame ++ )

	{
		int *p_soft_bits = p_soft_bitsN + frame*k*nPayloadSymbols;
		int *p_soft_bits_cache = p_soft_bits_cacheN + frame*k*nPayloadSymbols;

		p_soft_bits[j*nPayloadSymbols+index] = p_soft_bits_cache[index*k+j];	
	}
}