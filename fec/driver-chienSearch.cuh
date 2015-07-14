
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
	int tCapacity;
	int tMax;
	int MAXN;
	int L;

	// host
	int *powAlpha;
	int *lambda;
	int *el;
	int *ref_el;

	// device
	int *d_powAlpha;
	int *d_el, *d_lambda, *d_kk;

};
