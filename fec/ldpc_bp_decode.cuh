
#pragma once
#include <itpp/itcomm.h>

#include "ldpcFactory.h"
#include "dvb_types.h"

class ldpc_gpu
{
protected:
bool syndrome_check_gpu( ) ;

void updateVariableNode_gpu( ) ;

void updateCheckNode_gpu( );

void initializeMVC_gpu( );

bool check_parity_cpu(char *LLR);

public:
int bp_decode(int *LLRin, int *LLRout,
	bool psc = true,			//!< check syndrom after each iteration
	int max_iters = 50 );		//!< Maximum number of iterations

	/*!
	   * LDPC����   *
		* \param		LLRin	�������룺������Ȼ��ֵ����
		* \param		bitout	�������������������Ϣλ�����ַ�����
		* \param		psc	    �������룺�Ƿ�ÿ�ε�����������żУ��
		* \param		max_iters �������룺����������
	*/
int bp_decode_once(char *LLRout, int code_rate, 
	int *LLRin = NULL, 
	bool psc = true,			//!< check syndrom after each iteration
	int max_iters = 50 );		//!< Maximum number of iterations
int bp_decode_once(itpp::vec& softbits, char *LLRout, int code_rate );		//!< Maximum number of iterations
int bp_decode_once(double* softbits, char *LLRout, int code_rate );		//!< Maximum number of iterations


int decode_soft( scmplx* sym, double N0, int nPayloadSymbols, int M, int k,
	int *pFrame, int code_rate, 
	double* p_soft_bits, double* p_soft_bits_cache, 
	char* p_bitLDPC );	

	/*!
	   * LDPC��������ʼ��   *
		* \param	nvar 	�������룺�����ڵ���Ŀ�����볤��
		* \param	ncheck	�������룺У��ڵ���Ŀ����Ϣ����
		* \param	nmaxX1	�������룺У�����ÿ�С�1���ĸ��������ֵ
		* \param	nmaxX2  �������룺У�����ÿ�С�1���ĸ��������ֵ
		* \param	V 		�������룺У�����ÿ�С�1����������
		* \param	sumX1	�������룺У�����ÿ�С�1���ĸ���
		* \param	sumX2	�������룺У�����ÿ�С�1���ĸ���
		* \param	iind	�������룺У�����ÿ�С�1��������
		* \param	jind	�������룺У�����ÿ�С�1��������
		* \param	Dint1/2/3		�������룺ͬ������Ȼ��class LLR_calc_unit
		* \param	logexp_table	�������룺������Ȼ�Ȳ��ұ�
	*/
	bool	initialize( LDPC_CodeFactory* pcodes, scmplx* psymbols );
	void	updateSymbolsTemplate( scmplx* psymbols );

	~ldpc_gpu();

private:
	bool	release();
	float	distance( const scmplx& cL, const scmplx& cR );

private:
	int* d_synd ;
#if 0
	int* d_sumX1 ;
	int* d_sumX2 ;
	int* d_mcv ;
	int* d_mvc ;
	int* d_iind ;
	int* d_jind ;
	int* d_V ;

	int* d_logexp_table ;
#endif
	//int *d_ml, *d_mr ;
	
	int* d_LLRin ;
	char* d_LLRout ;
	int* h_mcv, *h_mvc ;
	int nvar, ncheck;
#if 0
	int* h_V, *h_sumX2;

private:
	int nmaxX1, nmaxX2; // max(sumX1) max(sumX2)
	short int Dint1, Dint2, Dint3;	//! Decoder (lookup-table) parameters
#endif
	//int max_cnd;	//! Maximum check node degree that the class can handle
	int QLLR_MAX;

	LDPC_DataFactory_GPU	m_ldpcDataPool;
	LDPC_DATA_GPU	*m_ldpcCurrent;

	scmplx*	d_pSymbolsTemplate ;// [M_CONST_NUMBER][32]
	scmplx*	d_pSymbolsIn;		// [FRAME_SIZE_NORMAL]
	float*	d_pDist2 ;			// [FRAME_SIZE_NORMAL][32]

	scmplx*	m_pSymbolsTemplate ;// [M_CONST_NUMBER][32]
	float*	m_pDist2 ;			// [FRAME_SIZE_NORMAL][32]

	int*	d_pSoftBitCache ;	// [FRAME_SIZE_NORMAL]
	int*	h_LLRin;
};