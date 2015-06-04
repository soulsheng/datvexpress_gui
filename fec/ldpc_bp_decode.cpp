
#include "ldpc_bp_decode.h"
#include <limits>
#include <iostream>
#include <sstream>
using namespace std;
#include "helper_timer.h"
//#include "driverUtility.h"
#include "dvbUtility.h"

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
		switch (sumX2[j]) {
		case 0:
			cout << "LDPC_Code::bp_decode(): sumX2[j]=0" << endl;
		case 1:
			cout << "LDPC_Code::bp_decode(): sumX2[j]=1" << endl;
		case 2: {
			mcv[j+ncheck] = mvc[jind[j]];
			mcv[j] = mvc[jind[j+ncheck]];
			break;
				}
		case 3: {
			int j0 = j;
			int m0 = mvc[jind[j0]];
			int j1 = j0 + ncheck;
			int m1 = mvc[jind[j1]];
			int j2 = j1 + ncheck;
			int m2 = mvc[jind[j2]];
			mcv[j0] = Boxplus(m1, m2, Dint1, Dint2, Dint3, logexp_table);
			mcv[j1] = Boxplus(m0, m2, Dint1, Dint2, Dint3, logexp_table);
			mcv[j2] = Boxplus(m0, m1, Dint1, Dint2, Dint3, logexp_table);
			break;
				}
		case 4: {
			int j0 = j;
			int m0 = mvc[jind[j0]];
			int j1 = j0 + ncheck;
			int m1 = mvc[jind[j1]];
			int j2 = j1 + ncheck;
			int m2 = mvc[jind[j2]];
			int j3 = j2 + ncheck;
			int m3 = mvc[jind[j3]];
			int m01 = Boxplus(m0, m1, Dint1, Dint2, Dint3, logexp_table);
			int m23 = Boxplus(m2, m3, Dint1, Dint2, Dint3, logexp_table);
			mcv[j0] = Boxplus(m1, m23, Dint1, Dint2, Dint3, logexp_table);
			mcv[j1] = Boxplus(m0, m23, Dint1, Dint2, Dint3, logexp_table);
			mcv[j2] = Boxplus(m01, m3, Dint1, Dint2, Dint3, logexp_table);
			mcv[j3] = Boxplus(m01, m2, Dint1, Dint2, Dint3, logexp_table);
			break;
				}
		case 5: {
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
			int m01 = Boxplus(m0, m1, Dint1, Dint2, Dint3, logexp_table);
			int m02 = Boxplus(m01, m2, Dint1, Dint2, Dint3, logexp_table);
			int m34 = Boxplus(m3, m4, Dint1, Dint2, Dint3, logexp_table);
			int m24 = Boxplus(m2, m34, Dint1, Dint2, Dint3, logexp_table);
			mcv[j0] = Boxplus(m1, m24, Dint1, Dint2, Dint3, logexp_table);
			mcv[j1] = Boxplus(m0, m24, Dint1, Dint2, Dint3, logexp_table);
			mcv[j2] = Boxplus(m01, m34, Dint1, Dint2, Dint3, logexp_table);
			mcv[j3] = Boxplus(m02, m4, Dint1, Dint2, Dint3, logexp_table);
			mcv[j4] = Boxplus(m02, m3, Dint1, Dint2, Dint3, logexp_table);
			break;
				}
		case 6: {
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
			int m01 = Boxplus(m0, m1, Dint1, Dint2, Dint3, logexp_table);
			int m23 = Boxplus(m2, m3, Dint1, Dint2, Dint3, logexp_table);
			int m45 = Boxplus(m4, m5, Dint1, Dint2, Dint3, logexp_table);
			int m03 = Boxplus(m01, m23, Dint1, Dint2, Dint3, logexp_table);
			int m25 = Boxplus(m23, m45, Dint1, Dint2, Dint3, logexp_table);
			int m0145 = Boxplus(m01, m45, Dint1, Dint2, Dint3, logexp_table);
			mcv[j0] = Boxplus(m1, m25, Dint1, Dint2, Dint3, logexp_table);
			mcv[j1] = Boxplus(m0, m25, Dint1, Dint2, Dint3, logexp_table);
			mcv[j2] = Boxplus(m0145, m3, Dint1, Dint2, Dint3, logexp_table);
			mcv[j3] = Boxplus(m0145, m2, Dint1, Dint2, Dint3, logexp_table);
			mcv[j4] = Boxplus(m03, m5, Dint1, Dint2, Dint3, logexp_table);
			mcv[j5] = Boxplus(m03, m4, Dint1, Dint2, Dint3, logexp_table);
			break;
				}
		default: {
			int nodes = sumX2[j];

			nodes--;
			jj[0] = j;
			m[0] = mvc[jind[jj[0]]];
			for(int i = 1; i <= nodes; i++ ) {
				jj[i] = jj[i-1] + ncheck;
				m[i] = mvc[jind[jj[i]]];
			}

			// compute partial sums from the left and from the right
			ml[0] = m[0];
			mr[0] = m[nodes];
			for(int i = 1; i < nodes; i++ ) {
				ml[i] = Boxplus( ml[i-1], m[i], Dint1, Dint2, Dint3, logexp_table );
				mr[i] = Boxplus( mr[i-1], m[nodes-i], Dint1, Dint2, Dint3, logexp_table );
			}

			// merge partial sums
			mcv[jj[0]] = mr[nodes-1];
			mcv[jj[nodes]] = ml[nodes-1];
			for(int i = 1; i < nodes; i++ )
				mcv[jj[i]] = Boxplus( ml[i-1], mr[nodes-1-i], Dint1, Dint2, Dint3, logexp_table );
				 }
		}  // switch statement
	}
}

void ldpc_decoder::updateVariableNode( int nvar, int* sumX1, int* mcv, int* mvc, int* iind, int * LLRin, char * LLRout ) 
{
	for (int i = 0; i < nvar; i++) {
		
		int mvc_temp = LLRin[i];

		switch (sumX1[i]) {
		case 0:
			cout << "LDPC_Code::bp_decode(): sumX1[i]=0" << endl;
		case 1: {
			/* This case is rare but apparently occurs for codes used in
			the DVB-T2 standard.
			*/
			int m0 = mcv[iind[i]];
			mvc[i] = LLRin[i];
			mvc_temp = LLRin[i] + m0;
			break;
				}
		case 2: {
			int m0 = mcv[iind[i]];
			int i1 = i + nvar;
			int m1 = mcv[iind[i1]];
			mvc[i] = LLRin[i] + m1;
			mvc[i1] = LLRin[i] + m0;
			mvc_temp = mvc[i1] + m1;
			break;
				}
		case 3: {
			int i0 = i;
			int m0 = mcv[iind[i0]];
			int i1 = i0 + nvar;
			int m1 = mcv[iind[i1]];
			int i2 = i1 + nvar;
			int m2 = mcv[iind[i2]];
			mvc_temp = LLRin[i] + m0 + m1 + m2;
			mvc[i0] = mvc_temp - m0;
			mvc[i1] = mvc_temp - m1;
			mvc[i2] = mvc_temp - m2;
			break;
				}
		case 4: {
			int i0 = i;
			int m0 = mcv[iind[i0]];
			int i1 = i0 + nvar;
			int m1 = mcv[iind[i1]];
			int i2 = i1 + nvar;
			int m2 = mcv[iind[i2]];
			int i3 = i2 + nvar;
			int m3 = mcv[iind[i3]];
			mvc_temp = LLRin[i] + m0 + m1 + m2 + m3;
			mvc[i0] = mvc_temp - m0;
			mvc[i1] = mvc_temp - m1;
			mvc[i2] = mvc_temp - m2;
			mvc[i3] = mvc_temp - m3;
			break;
				}
		default:   { // differential update
			int index_iind = i; // tracks i+jp*nvar
			for (int jp = 0; jp < sumX1[i]; jp++) {
				mvc_temp +=  mcv[iind[index_iind]];
				index_iind += nvar;
			}
			index_iind = i;  // tracks i+j*nvar
			for (int j = 0; j < sumX1[i]; j++) {
				mvc[index_iind] = mvc_temp - mcv[iind[index_iind]];
				index_iind += nvar;
			}
				   }
		}

		LLRout[i] = mvc_temp<0;

	}
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

int ldpc_decoder::bp_decode(int *LLRin, char *LLRout)		//!< Maximum number of iterations
{
	this->LLRin = LLRin; 
	this->LLRout = LLRout;

	StopWatchInterface	*timerStep;
	sdkCreateTimer( &timerStep );
	vector<float>	timerStepValue( (max_iters+1)*3 );

  // initial step
	memset( mvc, 0, nvar * nmaxX1 * sizeof(int) );
	memset( mcv, 0, ncheck * nmaxX2 * sizeof(int) );
	initializeMVC(nvar, sumX1, mvc, LLRin);

#if WRITE_FILE_FOR_DRIVER
	static bool bRunOnce1 = false;
	if( !bRunOnce1 ){
		writeArray( mvc, nvar * nmaxX1, "../data/mvcInit.txt" );		
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

	updateCheckNode(ncheck, sumX2, mcv, mvc, jind, Dint1, Dint2, Dint3, logexp_table );

	sdkStopTimer( &timerStep );
	timerStepValue[iter*3] = sdkGetTimerValue( &timerStep );

#if WRITE_FILE_FOR_DRIVER
	static bool bRunOnce1 = false;
	if( iter == 1 && !bRunOnce1 ){

		writeArray( mcv, ncheck * nmaxX2, "../data/mcv.txt" );

		bRunOnce1 = true;
	}
#endif

	sdkResetTimer( &timerStep );
	sdkStartTimer( &timerStep );
 
    // step 2: variable to check nodes
	updateVariableNode(nvar, sumX1, mcv, mvc, iind, LLRin, LLRout);

	sdkStopTimer( &timerStep );
	timerStepValue[iter*3+1] = sdkGetTimerValue( &timerStep );

#if WRITE_FILE_FOR_DRIVER
	static bool bRunOnce2 = false;
	if( iter == 1 && !bRunOnce2 ){

		writeArray( LLRout, nvar, "../data/output.txt" );
		writeArray( mvc, nvar * nmaxX1, "../data/mvc.txt" );		

		bRunOnce2 = true;
	}
#endif

	sdkResetTimer( &timerStep );
	sdkStartTimer( &timerStep );

	if (psc && syndrome_check(LLRout, ncheck, sumX2, V)) {
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

int ldpc_decoder::bp_decode( vec& softbits, char *LLRout )
{
	QLLRvec llrIn = ldpc.get_llrcalc().to_qllr(softbits);

	return bp_decode( llrIn._data(), LLRout);	
}

int ldpc_decoder::bp_decode( double* softbits, char *LLRout )
{
	vec  softVec( nvar );
	convertBufferToVec( softbits, softVec );
	return bp_decode( softVec, LLRout );
}

void ldpc_decoder::initialize(
	bool psc /*= true*/, /*!< check syndrom after each iteration */ 
	int max_iters /*= 50 */ )
{
	LDPC_Generator_Systematic G; // for codes created with ldpc_gen_codes since generator exists
	
	ldpc.load_code(FILENAME_IT34, &G);


	int nmaxX1 = max(ldpc.sumX1._data(), ldpc.sumX1.size());
	int nmaxX2 = max(ldpc.sumX2._data(), ldpc.sumX2.size());
	int nminX1 = min(ldpc.sumX1._data(), ldpc.sumX1.size());
	int nminX2 = min(ldpc.sumX2._data(), ldpc.sumX2.size());

	int nmaxI = max(ldpc.iind._data(), ldpc.iind.size());
	int nmaxJ = max(ldpc.jind._data(), ldpc.jind.size());
	int nminI = min(ldpc.iind._data(), ldpc.iind.size());
	int nminJ = min(ldpc.jind._data(), ldpc.jind.size());

#if 1
	cout << "max(iind) = " << nmaxI << endl;// max(iind) = nvar*nmaxX1-1
	cout << "max(jind) = " << nmaxJ << endl;// max(jind) = nvar*nmaxX1-1
	cout << "min(iind) = " << nminI << endl;// min(iind) = 0
	cout << "min(jind) = " << nminJ << endl;// min(jind) = 0

	cout << "ldpc.nvar = " << ldpc.nvar << endl;		// nvar = 16200
	cout << "ldpc.ncheck = " << ldpc.ncheck << endl;	// ncheck = 8100//8073 
	cout << "ldpc.sumX1.size() = " << ldpc.sumX1.size() << endl;	// = nvar
	cout << "ldpc.sumX2.size() = " << ldpc.sumX2.size() << endl;	// = ncheck
	cout << "max(sumX1) = " << nmaxX1 << endl;// max(sumX1) = 3//19
	cout << "max(sumX2) = " << nmaxX2 << endl;// max(sumX2) = 6//10
	cout << "min(sumX1) = " << nminX1 << endl;// min(sumX1) = 3//2
	cout << "min(sumX2) = " << nminX2 << endl;// min(sumX2) = 6//7
	cout << "ldpc.V.size() = " << ldpc.V.size() << endl;			// = ncheck * max(sumX2)
	cout << "ldpc.iind.size() = " << ldpc.iind.size() << endl;		// = nvar * max(sumX1)
	cout << "ldpc.jind.size() = " << ldpc.jind.size() << endl;		// = ncheck * max(sumX2)

	cout << "ldpc.mvc.size() = " << ldpc.mvc.size() << endl;		// = nvar * max(sumX1)
	cout << "ldpc.mcv.size() = " << ldpc.mcv.size() << endl;		// = ncheck * max(sumX2)

	cout << "ldpc.llrcalc.Dint1 = " << ldpc.llrcalc.Dint1 << endl;	// Dint1 = 12
	cout << "ldpc.llrcalc.Dint2 = " << ldpc.llrcalc.Dint2 << endl;	// Dint2 = 300
	cout << "ldpc.llrcalc.Dint3 = " << ldpc.llrcalc.Dint3 << endl;	// Dint3 = 7

	cout << "ldpc.llrcalc.logexp_table.size() = " << ldpc.llrcalc.logexp_table.size() << endl;// = 300
#endif

	this->nvar = ldpc.nvar;
	this->ncheck = ldpc.ncheck;
	this->nmaxX1 = nmaxX1;
	this->nmaxX2 = nmaxX2; // max(sumX1) max(sumX2)
	this->V = ldpc.V._data();
	this->sumX1 = ldpc.sumX1._data();
	this->sumX2 = ldpc.sumX2._data();
	this->iind = ldpc.iind._data();
	this->jind = ldpc.jind._data();	// Parity check matrix parameterization
	this->mvc = ldpc.mvc._data(); 
	this->mcv = ldpc.mcv._data();	// temporary storage for decoder (memory allocated when codec defined)
	this->Dint1 = ldpc.llrcalc.Dint1;
	this->Dint2 = ldpc.llrcalc.Dint2;
	this->Dint3 = ldpc.llrcalc.Dint3;	//! Decoder (lookup-table) parameters
	this->logexp_table = ldpc.llrcalc.logexp_table._data();		//! The lookup tables for the decoder
	this->psc = psc;			//!< check syndrom after each iteration
	this->max_iters = max_iters;
}

float ldpc_decoder::get_rate()
{
	return ldpc.get_rate();
}
