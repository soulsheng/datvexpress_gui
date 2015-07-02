
#pragma	once


class	driverUpdataVar
{

public:
	bool	launch();		//!	launch kernel 

	bool	verify();	//! verify whether the result is right or wrong
	
	driverUpdataVar( );

	~driverUpdataVar();

private:
	int nvar, ncheck;
	int nmaxX1, nmaxX2;

	// host
	int *sumX1;
	int *iind;
	int *mvc, *mcv;
	int *input;	char *output;

	int *ref_mvc;	char *ref_output;

	// device
	int *d_sumX1;
	int *d_iind;
	int *d_mvc, *d_mcv;
	int *d_input;	char *d_output;

};
