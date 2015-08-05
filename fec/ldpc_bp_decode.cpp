
#include "ldpc_bp_decode.h"
#include <limits>
#include <iostream>
#include <sstream>
using namespace std;
#include "helper_timer.h"
//#include "driverUtility.h"
#include "dvbUtility.h"
#include "itppUtility.h"

#define		SIZE_BLOCK			64
#define		USE_BLOCK			0

bool ldpc_decoder::syndrome_check(char *LLR,
	int ncheck, 
	int* sumX2, 
	int* V ) 
{
	// Please note the IT++ convention that a sure zero corresponds to
	// LLR=+infinity
	int i, j, synd, vi;

	for (j = 0; j < ncheck; j++) {
		synd = 0;
		int vind = j; // tracks j+i*ncheck
		for (i = 0; i < sumX2[j]; i++) {
			vi = V[vind];
			if (LLR[vi]) {
				synd++;
			}
			vind += ncheck;
		}
		if ((synd&1) == 1) {
			return false;  // codeword is invalid
		}
	}
	return true;   // codeword is valid
}

int  ldpc_decoder::logexp(int x,
	short int Dint1, short int Dint2, short int Dint3,	//! Decoder (lookup-table) parameters
	int* logexp_table )		//! The lookup tables for the decoder
{
	int ind = x >> Dint3;
	if (ind >= Dint2) // outside table
		return 0;

	// Without interpolation
	return logexp_table[ind];
}

int ldpc_decoder::Boxplus(int a, int b,
	short int Dint1, short int Dint2, short int Dint3,	//! Decoder (lookup-table) parameters
	int* logexp_table )		//! The lookup tables for the decoder
{
	int a_abs = (a > 0 ? a : -a);
	int b_abs = (b > 0 ? b : -b);
	int minabs = (a_abs > b_abs ? b_abs : a_abs);
	int term1 = (a > 0 ? (b > 0 ? minabs : -minabs)
		: (b > 0 ? -minabs : minabs));

	const int QLLR_MAX = (1<<31 -1)>>4;//(std::numeric_limits<int>::max() >> 4);

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
	int term2 = logexp((apb > 0 ? apb : -apb), Dint1, Dint2, Dint3, logexp_table);
	int amb = a - b;
	int term3 = logexp((amb > 0 ? amb : -amb), Dint1, Dint2, Dint3, logexp_table);
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

void ldpc_decoder::updateCheckNode( int ncheck, int* sumX2, int* mcv, int* mvc, int* jind, short int Dint1, short int Dint2, short int Dint3, int* logexp_table ) 
{

	//! Maximum check node degree that the class can handle
	static const int max_cnd = 200;

	// allocate temporary variables used for the check node update
	int jj[max_cnd];
	int m[max_cnd];
	int ml[max_cnd];
	int mr[max_cnd];


	for (int j = 0; j < ncheck; j++) {
		// The check node update calculations are hardcoded for degrees
		// up to 6.  For larger degrees, a general algorithm is used.

			int nodes = sumX2[j];

			nodes--;

			for(int i = 0; i <= nodes; i++ ) {
				m[i] = mvc[jind[j+i*ncheck]];
			}

			// compute partial sums from the left and from the right
			ml[0] = m[0];
			mr[0] = m[nodes];
			for(int i = 1; i < nodes; i++ ) {
				ml[i] = Boxplus( ml[i-1], m[i], Dint1, Dint2, Dint3, logexp_table );
				mr[i] = Boxplus( mr[i-1], m[nodes-i], Dint1, Dint2, Dint3, logexp_table );
			}

			// merge partial sums
			mcv[j] = mr[nodes-1];
			mcv[j+nodes*ncheck] = ml[nodes-1];
			for(int i = 1; i < nodes; i++ )
				mcv[j+i*ncheck] = Boxplus( ml[i-1], mr[nodes-1-i], Dint1, Dint2, Dint3, logexp_table );

		
	}
}

void ldpc_decoder::updateVariableNode( int nvar, int* sumX1, int* mcv, int* mvc, int* iind, int * LLRin, char * LLRout ) 
{
#if  USE_BLOCK
	int block = SIZE_BLOCK ;
	int grid = (nvar + block - 1) / block ;

	for (int g = 0; g < grid; g++) {
		unsigned int nMin = (unsigned int)(-1);
		unsigned int nMax = 0;
	for (int t = 0; t < block; t++ ) {
		int i = block * g + t;
		
		if( i >= nvar )
			break;

		int mvc_temp = LLRin[i];

		for (int jp = 0; jp < sumX1[i]; jp++) {
			mvc_temp +=  mcv[iind[ i + jp*nvar]];// iind[ i + jp*nvar]  (0~48k)
			//printf( "%d ", iind[ i + jp*nvar] );
			if( iind[ i + jp*nvar] > nMax )
				nMax = iind[ i + jp*nvar];
			if( iind[ i + jp*nvar] < nMin )
				nMin = iind[ i + jp*nvar];

		}

		for (int j = 0; j < sumX1[i]; j++) {
			mvc[i + j*nvar] = mvc_temp - mcv[iind[i + j*nvar]];
		}
		

		LLRout[i] = mvc_temp<0;
	}
		printf( "block %d, (%d ~ %d, %d) \n", g, nMin, nMax, nMax-nMin );
		//system("pause");
	}
#else
	for (int i = 0; i < nvar; i++) {
		
		int mvc_temp = LLRin[i];

		for (int jp = 0; jp < sumX1[i]; jp++) {
			mvc_temp +=  mcv[iind[ i + jp*nvar]];
		}

		for (int j = 0; j < sumX1[i]; j++) {
			mvc[i + j*nvar] = mvc_temp - mcv[iind[i + j*nvar]];
		}
		

		LLRout[i] = mvc_temp<0;

	}
#endif
}

void ldpc_decoder::initializeMVC( int nvar, int* sumX1, int* mvc, int * LLRin ) 
{
	for (int i = 0; i < nvar; i++) {
		int index = i;
		for (int j = 0; j < sumX1[i]; j++) {
			mvc[index] = LLRin[i];
			index += nvar;
		}
	}
}

int ldpc_decoder::bp_decode(int *LLRin, char *LLRout, int code_rate)		//!< Maximum number of iterations
{
	this->LLRin = LLRin; 
	this->LLRout = LLRout;

	m_ldpcCurrent = m_ldpcPool.findLDPC_DATA( code_rate );

	StopWatchInterface	*timerStep;
	sdkCreateTimer( &timerStep );
	vector<float>	timerStepValue( (max_iters+1)*3 );

  // initial step
	memset( m_ldpcCurrent->mvc, 0, m_ldpcCurrent->nvar * m_ldpcCurrent->nmaxX1 * sizeof(int) );
	memset( m_ldpcCurrent->mcv, 0, m_ldpcCurrent->ncheck * m_ldpcCurrent->nmaxX2 * sizeof(int) );
	initializeMVC(m_ldpcCurrent->nvar, m_ldpcCurrent->sumX1, m_ldpcCurrent->mvc, LLRin);

#if WRITE_FILE_FOR_DRIVER
	static bool bRunOnce1 = false;
	if( !bRunOnce1 ){
		writeFile( m_ldpcCurrent->nvar, m_ldpcCurrent->ncheck, m_ldpcCurrent->nmaxX1, m_ldpcCurrent->nmaxX2, "../data/ldpcSize.txt" );
		writeArray( m_ldpcCurrent->mvc, m_ldpcCurrent->nvar * m_ldpcCurrent->nmaxX1, "../data/mvcInit.txt" );		
		
		writeArray( LLRin, m_ldpcCurrent->nvar, "../data/input.txt" );
		writeArray( m_ldpcCurrent->sumX1, m_ldpcCurrent->nvar, "../data/sumX1.txt" );
		writeArray( m_ldpcCurrent->sumX2, m_ldpcCurrent->ncheck, "../data/sumX2.txt" );

		writeArray( m_ldpcCurrent->iind, m_ldpcCurrent->nvar * m_ldpcCurrent->nmaxX1, "../data/iind.txt" );
		writeArray( m_ldpcCurrent->jind, m_ldpcCurrent->ncheck * m_ldpcCurrent->nmaxX2, "../data/jind.txt" );
		writeArray( m_ldpcCurrent->logexp_table, m_ldpcCurrent->Dint2, "../data/logexp.txt" );

		bRunOnce1 = true;
	}
#endif

  bool is_valid_codeword = false;
  int iter = 0;
  do {
    iter++;
    //if (nvar >= 100000) { it_info_no_endl_debug("."); }
    // --------- Step 1: check to variable nodes ----------
	sdkResetTimer( &timerStep );
	sdkStartTimer( &timerStep );

	updateCheckNode(m_ldpcCurrent->ncheck, m_ldpcCurrent->sumX2, 
		m_ldpcCurrent->mcv, m_ldpcCurrent->mvc, 
		m_ldpcCurrent->jind, 
		m_ldpcCurrent->Dint1, m_ldpcCurrent->Dint2, m_ldpcCurrent->Dint3, 
		m_ldpcCurrent->logexp_table );

	sdkStopTimer( &timerStep );
	timerStepValue[iter*3] = sdkGetTimerValue( &timerStep );

#if WRITE_FILE_FOR_DRIVER
	static bool bRunOnce1 = false;
	if( iter == 1 && !bRunOnce1 ){

		writeArray( m_ldpcCurrent->mcv, m_ldpcCurrent->ncheck * m_ldpcCurrent->nmaxX2, "../data/mcv.txt" );

		bRunOnce1 = true;
	}
#endif

	sdkResetTimer( &timerStep );
	sdkStartTimer( &timerStep );
 
    // step 2: variable to check nodes
	updateVariableNode(m_ldpcCurrent->nvar, m_ldpcCurrent->sumX1, 
		m_ldpcCurrent->mcv, m_ldpcCurrent->mvc, 
		m_ldpcCurrent->iind, 
		LLRin, LLRout);

	sdkStopTimer( &timerStep );
	timerStepValue[iter*3+1] = sdkGetTimerValue( &timerStep );

#if WRITE_FILE_FOR_DRIVER
	static bool bRunOnce2 = false;
	if( iter == 1 && !bRunOnce2 ){

		writeArray( LLRout, m_ldpcCurrent->nvar, "../data/output.txt" );
		writeArray( m_ldpcCurrent->mvc, m_ldpcCurrent->nvar * m_ldpcCurrent->nmaxX1, "../data/mvc.txt" );		

		bRunOnce2 = true;
	}
#endif

	sdkResetTimer( &timerStep );
	sdkStartTimer( &timerStep );

	if (psc && syndrome_check(LLRout, m_ldpcCurrent->ncheck, m_ldpcCurrent->sumX2, m_ldpcCurrent->V)) {
	  is_valid_codeword = true;
      break;
    }

	sdkStopTimer( &timerStep );
	timerStepValue[iter*3+2] = sdkGetTimerValue( &timerStep );

  }
  while (iter < max_iters);

  for (int i=1;i<iter*3;i++)
  {
	//  cout  << "timerStepValue[ " << i << " ] = "<< timerStepValue[i] << " ms, " << endl;
  }
  cout << endl << endl ;
  sdkDeleteTimer( &timerStep );

  return (is_valid_codeword ? iter : -iter);
}

int ldpc_decoder::bp_decode( vec& softbits, char *LLRout, int code_rate )
{
	m_ldpcCurrent = m_ldpcPool.findLDPC_DATA( code_rate );
	QLLRvec llrIn = m_ldpcCurrent->getCode()->get_llrcalc().to_qllr(softbits);

	return bp_decode( llrIn._data(), LLRout, code_rate);	
}

int ldpc_decoder::bp_decode( double* softbits, char *LLRout, int code_rate )
{
	m_ldpcCurrent = m_ldpcPool.findLDPC_DATA( code_rate );
	vec  softVec( m_ldpcCurrent->nvar );
	convertBufferToVec( softbits, softVec );
	return bp_decode( softVec, LLRout, code_rate );
}

void ldpc_decoder::initialize(LDPC_CodeFactory* pcodes,
	bool psc /*= true*/, /*!< check syndrom after each iteration */ 
	int max_iters /*= 50 */ )
{
	m_ldpcPool.initialize(pcodes);

	this->psc = psc;			//!< check syndrom after each iteration
	this->max_iters = max_iters;
}

