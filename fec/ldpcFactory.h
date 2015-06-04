
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

	LDPC_DATA( LDPC_Code* pCode );

	LDPC_Code* pldpc;
	int get_nvar() const { return nvar; }
	int get_ncheck() const { return ncheck; }
	int get_ninfo() const { return nvar - ncheck; }
	float get_rate();
	LDPC_Code* getCode()	{ return pldpc;}
};

struct LDPC_DATA_GPU
{
	int nvar, ncheck;
	int nmaxX1, nmaxX2; // max(sumX1) max(sumX2)

	int* d_sumX1 ;
	int* d_sumX2 ;
	int* d_mcv ;
	int* d_mvc ;
	int* d_iind ;
	int* d_jind ;
	int* d_V ;

	int* d_logexp_table ;

	short int Dint1, Dint2, Dint3;	//! Decoder (lookup-table) parameters

	int* h_V, *h_sumX2;
	int* h_mcv, *h_mvc ;

	LDPC_DATA_GPU( LDPC_Code* pCode );
	~LDPC_DATA_GPU();

	LDPC_Code* pldpc;
	int get_nvar() const { return nvar; }
	int get_ncheck() const { return ncheck; }
	int get_ninfo() const { return nvar - ncheck; }
	float get_rate();
	LDPC_Code* getCode()	{ return pldpc;}

};


class LDPC_CodeFactory
{
public:
	LDPC_CodeFactory();
	~LDPC_CodeFactory();

	/*! 搜索解调器，QPSK/8PSK/16APSK/32APSK 
		* \param 	modType 		参数输入：从BB Header解析的调制类型
		* \return 	从调试器工厂查找匹配的解调器
	*/
	LDPC_Code* findLDPC_Code(int code_rate);
	void	initialize();

protected:
private:

	typedef map<int, LDPC_Code*> LDPC_CodePool;
	typedef map<int, LDPC_Code*>::iterator LDPC_CodePoolItr;
	typedef pair<int, LDPC_Code*> LDPC_CodePoolPair;

	LDPC_CodePool	m_LDPC_CodePool;
};

class LDPC_DataFactory
{
public:
	LDPC_DataFactory();
	~LDPC_DataFactory();

	/*! 搜索解调器，QPSK/8PSK/16APSK/32APSK 
		* \param 	modType 		参数输入：从BB Header解析的调制类型
		* \return 	从调试器工厂查找匹配的解调器
	*/
	LDPC_DATA* findLDPC_DATA(int code_rate);
	void	initialize(LDPC_CodeFactory* pCodes);

protected:
private:
	typedef map<int, LDPC_DATA*> LDPCPool;
	typedef map<int, LDPC_DATA*>::iterator LDPCPoolItr;
	typedef pair<int, LDPC_DATA*> LDPCPoolPair;

	LDPCPool	m_LDPCPool;		//!解码器工厂，QPSK/8PSK/16APSK/32APSK
};

class LDPC_DataFactory_GPU
{
public:
	LDPC_DataFactory_GPU();
	~LDPC_DataFactory_GPU();

	/*! 搜索解调器，QPSK/8PSK/16APSK/32APSK 
		* \param 	modType 		参数输入：从BB Header解析的调制类型
		* \return 	从调试器工厂查找匹配的解调器
	*/
	LDPC_DATA_GPU* findLDPC_DATA(int code_rate);
	void	initialize(LDPC_CodeFactory* pCodes);

protected:
private:
	typedef map<int, LDPC_DATA_GPU*> LDPC_GPUPool;
	typedef map<int, LDPC_DATA_GPU*>::iterator LDPC_GPUPoolItr;
	typedef pair<int, LDPC_DATA_GPU*> LDPC_GPUPoolPair;

	LDPC_GPUPool	m_LDPCPool;		//!解码器工厂，QPSK/8PSK/16APSK/32APSK
};
