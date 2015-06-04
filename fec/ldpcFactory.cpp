
#include "ldpcFactory.h"
#include "dvbUtility.h"

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

LDPCFactory::LDPCFactory()
{
	
}

LDPCFactory::~LDPCFactory()
{
	for (LDPCPoolItr itr=m_LDPCPool.begin(); itr!=m_LDPCPool.end(); itr++)
		delete itr->second;

	m_LDPCPool.clear();
}

LDPC_DATA* LDPCFactory::findLDPC_DATA( int modType )
{
	LDPCPoolItr itr=m_LDPCPool.find( modType );
	if ( itr != m_LDPCPool.end() )
		return itr->second;
	else
		return NULL;
}

void LDPCFactory::initialize( )
{
	for (int i=0; i<CODE_RATE_COUNT; i++)
	{
		LDPC_DATA* pLDPC_DATA = new LDPC_DATA( i );

		m_LDPCPool.insert( LDPCPoolPair(i, pLDPC_DATA) );
	}
}

LDPC_DATA::LDPC_DATA( int code_rate )
{

	ifstream  testfile;
	testfile.open( g_filename_it[code_rate] );
	if ( testfile == NULL )
	{
		cout << "Can not find ldpc code file - \""
			<< g_filename_it[code_rate] << endl;
		return ;
	}
	testfile.close();

	LDPC_Generator_Systematic G; // for codes created with ldpc_gen_codes since generator exists

	ldpc.load_code(g_filename_it[code_rate], &G);


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
	return ldpc.get_rate();
}
