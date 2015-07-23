
#pragma	once


class	driverSoftBit
{

public:
	bool	launch();		//!	launch kernel 

	bool	verify();	//! verify whether the result is right or wrong
	
	driverSoftBit( );

	~driverSoftBit();

private:
	float N0;
	int k, M, Dint1, QLLR_MAX, n, nPayloadSymbols, nMulti;

	// host
	float *m_pDist2;
	int *m_pSoftBit;

	int *ref_pSoftBit;

	// device
	float *d_pDist2;
	int *d_pSoftBitCache;

};
