
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
	   * LDPC解码   *
		* \param		LLRin	数据输入：整形似然比值数组
		* \param		bitout	数据输出：解码后输出信息位——字符数组
		* \param		psc	    参数输入：是否每次迭代都进行奇偶校验
		* \param		max_iters 参数输入：最大迭代次数
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
	   * LDPC解码器初始化   *
		* \param	nvar 	参数输入：变量节点数目，编码长度
		* \param	ncheck	参数输入：校验节点数目，信息长度
		* \param	nmaxX1	参数输入：校验矩阵每列“1”的个数的最大值
		* \param	nmaxX2  参数输入：校验矩阵每行“1”的个数的最大值
		* \param	V 		参数输入：校验矩阵每行“1”的列坐标
		* \param	sumX1	参数输入：校验矩阵每列“1”的个数
		* \param	sumX2	参数输入：校验矩阵每行“1”的个数
		* \param	iind	参数输入：校验矩阵每行“1”的索引
		* \param	jind	参数输入：校验矩阵每行“1”的索引
		* \param	Dint1/2/3		参数输入：同对数似然比class LLR_calc_unit
		* \param	logexp_table	参数输入：对数似然比查找表
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