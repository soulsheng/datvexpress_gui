
#include "ldpcFactory.h"
#include "dvbUtility.h"
#ifdef USE_GPU
#include <cuda_runtime.h>
#endif
char	g_filename_it[CODE_RATE_COUNT][50] ={
"../data/dvbs2_r14.it",
"../data/dvbs2_r13.it",
"../data/dvbs2_r25.it",
"../data/dvbs2_r12.it",
"../data/dvbs2_r35.it",
"../data/dvbs2_r23.it",
"../data/dvbs2_r34.it",
"../data/dvbs2_r45.it",
"../data/dvbs2_r56.it",
"../data/dvbs2_r89.it"
,"../data/dvbs2_r910.it"
};

LDPC_DataFactory::LDPC_DataFactory()
{
	
}

LDPC_DataFactory::~LDPC_DataFactory()
{
	for (LDPCPoolItr itr=m_LDPCPool.begin(); itr!=m_LDPCPool.end(); itr++)
		delete itr->second;

	m_LDPCPool.clear();
}

LDPC_DATA* LDPC_DataFactory::findLDPC_DATA( int code_rate )
{
	LDPCPoolItr itr=m_LDPCPool.find( code_rate );
	if ( itr != m_LDPCPool.end() )
		return itr->second;
	else
		return NULL;
}

void LDPC_DataFactory::initialize( LDPC_CodeFactory* pCodes )
{
	for (int i=0; i<CODE_RATE_COUNT; i++)
	{
		LDPC_Code* pLDPC_Code = pCodes->findLDPC_Code(i);

		LDPC_DATA* pLDPC_DATA = new LDPC_DATA( pLDPC_Code );

		m_LDPCPool.insert( LDPCPoolPair(i, pLDPC_DATA) );
	}
}

LDPC_DATA::LDPC_DATA( LDPC_Code* pCode )
{
	pldpc = pCode;

	LDPC_Code& ldpc = *pldpc;

	int nmaxX1 = max(ldpc.sumX1._data(), ldpc.sumX1.size());
	int nmaxX2 = max(ldpc.sumX2._data(), ldpc.sumX2.size());
	int nminX1 = min(ldpc.sumX1._data(), ldpc.sumX1.size());
	int nminX2 = min(ldpc.sumX2._data(), ldpc.sumX2.size());

	int nmaxI = max(ldpc.iind._data(), ldpc.iind.size());
	int nmaxJ = max(ldpc.jind._data(), ldpc.jind.size());
	int nminI = min(ldpc.iind._data(), ldpc.iind.size());
	int nminJ = min(ldpc.jind._data(), ldpc.jind.size());

#if 0
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

}

float LDPC_DATA::get_rate()
{
	return getCode()->get_rate();
}

LDPC_CodeFactory::LDPC_CodeFactory()
{

}

LDPC_CodeFactory::~LDPC_CodeFactory()
{
	for (LDPC_CodePoolItr itr=m_LDPC_CodePool.begin(); itr!=m_LDPC_CodePool.end(); itr++)
		delete itr->second;

	m_LDPC_CodePool.clear();
}

LDPC_Code* LDPC_CodeFactory::findLDPC_Code( int code_rate )
{
	LDPC_CodePoolItr itr=m_LDPC_CodePool.find( code_rate );
	if ( itr != m_LDPC_CodePool.end() )
		return itr->second;
	else
		return NULL;
}

void LDPC_CodeFactory::initialize()
{
	for (int i=0; i<CODE_RATE_COUNT; i++)
	{
		ifstream  testfile;
		testfile.open( g_filename_it[i] );
		if ( testfile == NULL )
		{
			cout << "Can not find ldpc code file - \""
				<< g_filename_it[i] << endl;
			return ;
		}
		testfile.close();

		LDPC_Generator_Systematic G; // for codes created with ldpc_gen_codes since generator exists
		LDPC_Code* pLDPC_Code = new LDPC_Code( g_filename_it[i], &G );

		m_LDPC_CodePool.insert( LDPC_CodePoolPair(i, pLDPC_Code) );
	}
}


LDPC_DataFactory_GPU::LDPC_DataFactory_GPU()
{

}

LDPC_DataFactory_GPU::~LDPC_DataFactory_GPU()
{
	for (LDPC_GPUPoolItr itr=m_LDPCPool.begin(); itr!=m_LDPCPool.end(); itr++)
		delete itr->second;

	m_LDPCPool.clear();
}

LDPC_DATA_GPU* LDPC_DataFactory_GPU::findLDPC_DATA( int code_rate )
{
	LDPC_GPUPoolItr itr=m_LDPCPool.find( code_rate );
	if ( itr != m_LDPCPool.end() )
		return itr->second;
	else
		return NULL;
}

void LDPC_DataFactory_GPU::initialize( LDPC_CodeFactory* pCodes )
{
	for (int i=0; i<CODE_RATE_COUNT; i++)
	{
		LDPC_Code* pLDPC_Code = pCodes->findLDPC_Code(i);

		LDPC_DATA_GPU* pLDPC_DATA = new LDPC_DATA_GPU( pLDPC_Code );

		m_LDPCPool.insert( LDPC_GPUPoolPair(i, pLDPC_DATA) );
	}
}

LDPC_DATA_GPU::LDPC_DATA_GPU( LDPC_Code* pCode )
{
	pldpc = pCode;

	LDPC_Code& ldpc = *pldpc;

	int nmaxX1 = max(ldpc.sumX1._data(), ldpc.sumX1.size());
	int nmaxX2 = max(ldpc.sumX2._data(), ldpc.sumX2.size());
	int nminX1 = min(ldpc.sumX1._data(), ldpc.sumX1.size());
	int nminX2 = min(ldpc.sumX2._data(), ldpc.sumX2.size());

	int nmaxI = max(ldpc.iind._data(), ldpc.iind.size());
	int nmaxJ = max(ldpc.jind._data(), ldpc.jind.size());
	int nminI = min(ldpc.iind._data(), ldpc.iind.size());
	int nminJ = min(ldpc.jind._data(), ldpc.jind.size());

	this->nvar = ldpc.nvar;		this->ncheck = ldpc.ncheck;
	this->nmaxX1 = nmaxX1;	this->nmaxX2 = nmaxX2; // max(sumX1) max(sumX2)
	this->Dint1 = ldpc.llrcalc.Dint1;	
	this->Dint2 = ldpc.llrcalc.Dint2;	
	this->Dint3 = ldpc.llrcalc.Dint3;	//! Decoder (lookup-table) parameters

	this->h_V = ldpc.V._data();
	this->h_sumX2 = ldpc.sumX2._data();
#ifdef USE_GPU
	cudaMalloc( (void**)&d_sumX1, nvar * sizeof(int) );		// const 64 K
	cudaMemcpy( d_sumX1, ldpc.sumX1._data(), nvar * sizeof(int), cudaMemcpyHostToDevice );

	cudaMalloc( (void**)&d_sumX2, ncheck * sizeof(int) );	// const 32 K
	cudaMemcpy( d_sumX2, ldpc.sumX2._data(), ncheck * sizeof(int), cudaMemcpyHostToDevice );

	cudaMalloc( (void**)&d_iind, nvar * nmaxX1 * sizeof(int) );		// const 1.2 M
	cudaMemcpy( d_iind, ldpc.iind._data(), nvar * nmaxX1 * sizeof(int), cudaMemcpyHostToDevice );

	cudaMalloc( (void**)&d_jind, ncheck * nmaxX2 * sizeof(int) );	// const 300 K
	cudaMemcpy( d_jind, ldpc.jind._data(), ncheck * nmaxX2 * sizeof(int), cudaMemcpyHostToDevice );

	cudaMalloc( (void**)&d_V, ncheck * nmaxX2 * sizeof(int) );		// const 300 K
	cudaMemcpy( d_V, ldpc.V._data(), ncheck * nmaxX2 * sizeof(int), cudaMemcpyHostToDevice );

	cudaMalloc( (void**)&d_mcv, ncheck * nmaxX2 * sizeof(int) );
	cudaMemset( d_mcv, 0, ncheck * nmaxX2 * sizeof(int) );

	cudaMalloc( (void**)&d_mvc, nvar * nmaxX1 * sizeof(int) );
	cudaMemset( d_mvc, 0, nvar * nmaxX1 * sizeof(int) );

	cudaMalloc( (void**)&d_logexp_table, Dint2 * sizeof(int) );		// const 1.2 K
	cudaMemcpy( d_logexp_table, ldpc.llrcalc.logexp_table._data(), Dint2 * sizeof(int), cudaMemcpyHostToDevice );
#endif
}

LDPC_DATA_GPU::~LDPC_DATA_GPU()
{
#ifdef USE_GPU
	cudaFree( d_sumX1 );	cudaFree( d_sumX2 );

	cudaFree( d_iind );		cudaFree( d_jind );
	cudaFree( d_V );

	cudaFree( d_mcv );		cudaFree( d_mvc );

	cudaFree( d_logexp_table );	
#endif
}
