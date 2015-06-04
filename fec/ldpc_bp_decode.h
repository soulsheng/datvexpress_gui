
#pragma once
#include <itpp/itcomm.h>

#include "ldpcFactory.h"

using namespace std;
using namespace itpp;

class ldpc_decoder{
public:

bool syndrome_check(char* LLR,
	int ncheck, 
	int* sumX2, 
	int* V) ;

int  logexp(int x,
	short int Dint1, short int Dint2, short int Dint3,	//! Decoder (lookup-table) parameters
	int* logexp_table );		//! The lookup tables for the decoder

int Boxplus(int a, int b,
	short int Dint1, short int Dint2, short int Dint3,	//! Decoder (lookup-table) parameters
	int* logexp_table );		//! The lookup tables for the decoder

void initialize(LDPC_CodeFactory* pcodes,
	bool psc = true,			//!< check syndrom after each iteration
	int max_iters = 50 );		//!< Maximum number of iterations

void updateCheckNode( int ncheck, int* sumX2, 
	int* mcv, int* mvc, int* jind, 
	short int Dint1, short int Dint2, short int Dint3, 
	int* logexp_table );

void updateVariableNode( int nvar, int* sumX1, 
	int* mcv, int* mvc, int* iind, 
	int * LLRin, char * LLRout );

void initializeMVC( int nvar, int* sumX1, int* mvc, int * LLRin );

int bp_decode(int *LLRin, char *LLRout, int code_rate);
int bp_decode(vec& softbits, char *LLRout, int code_rate);		//!< Maximum number of iterations
int bp_decode(double* softbits, char *LLRout, int code_rate);		//!< Maximum number of iterations

public:
	

protected:
	int *LLRin; char *LLRout;

	LDPC_DataFactory	m_ldpcPool;
	LDPC_DATA	*m_ldpcCurrent;

	bool psc;			//!< check syndrom after each iteration
	int max_iters;
};