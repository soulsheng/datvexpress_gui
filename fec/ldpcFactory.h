
#pragma once

#include <itpp/itcomm.h>
using namespace itpp;

#include "modulatorDefinition.h"

struct LDPC_DATA
{
	int nvar, ncheck;
	int nmaxX1, nmaxX2; // max(sumX1) max(sumX2)
	int* V, * sumX1, * sumX2, * iind, * jind;	// Parity check matrix parameterization
	int* mvc; int* mcv;	// temporary storage for decoder (memory allocated when codec defined)
	short int Dint1, Dint2, Dint3;	//! Decoder (lookup-table) parameters
	int* logexp_table;		//! The lookup tables for the decoder

	LDPC_DATA( int code_rate );

	LDPC_Code ldpc;
	int get_nvar() const { return nvar; }
	int get_ncheck() const { return ncheck; }
	int get_ninfo() const { return nvar - ncheck; }
	float get_rate();
};


class LDPCFactory
{
public:
	LDPCFactory();
	~LDPCFactory();

	/*! 搜索解调器，QPSK/8PSK/16APSK/32APSK 
		* \param 	modType 		参数输入：从BB Header解析的调制类型
		* \return 	从调试器工厂查找匹配的解调器
	*/
	LDPC_DATA* findLDPC_DATA(int code_rate);
	void	initialize();

protected:
private:
	typedef map<int, LDPC_DATA*> LDPCPool;
	typedef map<int, LDPC_DATA*>::iterator LDPCPoolItr;
	typedef pair<int, LDPC_DATA*> LDPCPoolPair;

	LDPCPool	m_LDPCPool;		//!解码器工厂，QPSK/8PSK/16APSK/32APSK
};
