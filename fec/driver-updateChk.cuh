
#pragma	once


class	driverUpdataChk
{

public:
	bool	launch();		//!	launch kernel 

	bool	verify();	//! verify whether the result is right or wrong
	
	driverUpdataChk();

	~driverUpdataChk();

private:
	int nvar, ncheck;
	int nmaxX1, nmaxX2;
	short int Dint1, Dint2, Dint3;	//! Decoder (lookup-table) parameters
	int QLLR_MAX;

	// host
	int *sumX2;
	int *jind;
	int *mvc, *mcv;

	int *ref_mcv;

	int *logexp_table;

	// device
	int *d_sumX2;
	int *d_jind;
	int *d_mvc, *d_mcv;

	int* d_logexp_table ;

};
