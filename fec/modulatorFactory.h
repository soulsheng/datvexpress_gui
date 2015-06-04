
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
	int rate;
	int framesize;

	SymbolTable(int k, int rate=CR_3_4, int framesize=FRAME_NORMAL);
	cvec&	getSymbols()	{ return symbols;}
	ivec&	getBits10Symbols()	{ return bits10symbols;}
};


class ModulatorFactory
{
public:
	ModulatorFactory();
	~ModulatorFactory();

	/*! �����������QPSK/8PSK/16APSK/32APSK 
		* \param 	modType 		�������룺��BB Header�����ĵ�������
		* \return 	�ӵ�������������ƥ��Ľ����
	*/
	Modulator_2D* findModulator(int modType);

protected:
private:
	typedef map<int, Modulator_2D*> ModPool;
	typedef map<int, Modulator_2D*>::iterator ModPoolItr;
	typedef pair<int, Modulator_2D*> ModPoolPair;

	ModPool	m_modPool;		//!�����������QPSK/8PSK/16APSK/32APSK
};
