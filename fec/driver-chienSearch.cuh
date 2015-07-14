
#pragma	once


class	driverChienSearch
{

public:
	bool	launch();		//!	launch kernel 

	bool	verify();	//! verify whether the result is right or wrong
	
	driverChienSearch( );

	~driverChienSearch();

private:
	int m_nAlpha;
	int m_nCodeword;
	int m_nGrid;
	int MAXN;

	// host
	int *powAlpha;
	int *SCache;
	char* codeword;

	int *ref_SCache;

	// device
	int *d_powAlpha;
	int *d_SCache;          // Syndrome vector
	char* d_codeword;

};
