
#pragma once
#include <itpp/itcomm.h>

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
int bp_decode_once(int *LLRin, char *LLRout,
	bool psc = true,			//!< check syndrom after each iteration
	int max_iters = 50 );		//!< Maximum number of iterations
int bp_decode_once(itpp::vec& softbits, char *LLRout);		//!< Maximum number of iterations
int bp_decode_once(double* softbits, char *LLRout);		//!< Maximum number of iterations

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
	bool	initialize( );

	~ldpc_gpu();

	int get_nvar() const { return nvar; }
	int get_ncheck() const { return ncheck; }
	int get_ninfo() const { return nvar - ncheck; }
	float get_rate();

private:
	bool	release();

private:
	int* d_synd ;

	int* d_sumX1 ;
	int* d_sumX2 ;
	int* d_mcv ;
	int* d_mvc ;
	int* d_iind ;
	int* d_jind ;
	int* d_V ;

	int* d_logexp_table ;
	
	//int *d_ml, *d_mr ;
	
	int* d_LLRin ;
	char* d_LLRout ;
	
	int* h_V, *h_sumX2;
	int* h_mcv, *h_mvc ;

private:
	int nvar, ncheck;
	int nmaxX1, nmaxX2; // max(sumX1) max(sumX2)
	short int Dint1, Dint2, Dint3;	//! Decoder (lookup-table) parameters
	//int max_cnd;	//! Maximum check node degree that the class can handle
	int QLLR_MAX;

	itpp::LDPC_Code ldpc;

};