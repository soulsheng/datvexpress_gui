
#pragma once

#include <itpp/itcomm.h>
using namespace itpp;

#include "modulatorDefinition.h"

struct SymbolTable
{
	cvec	symbols;
	ivec	bits10symbols;
	Vec<bvec>	bits2symbols;
	int		k;
	double  r1, r2, r3;
	CODE_RATE rate;
	FRAME_TYPE framesize;

	SymbolTable(int k, CODE_RATE rate=C3_4, FRAME_TYPE framesize=FECFRAME_NORMAL);
	cvec&	getSymbols()	{ return symbols;}
	ivec&	getBits10Symbols()	{ return bits10symbols;}
};


class ModulatorFactory
{
public:
	ModulatorFactory();
	~ModulatorFactory();

	/*! 搜索解调器，QPSK/8PSK/16APSK/32APSK 
		* \param 	modType 		参数输入：从BB Header解析的调制类型
		* \return 	从调试器工厂查找匹配的解调器
	*/
	Modulator_2D* findModulator(MOD_TYPE modType);

protected:
private:
	typedef map<MOD_TYPE, Modulator_2D*> ModPool;
	typedef map<MOD_TYPE, Modulator_2D*>::iterator ModPoolItr;
	typedef pair<MOD_TYPE, Modulator_2D*> ModPoolPair;

	ModPool	m_modPool;		//!解调器工厂，QPSK/8PSK/16APSK/32APSK
};
