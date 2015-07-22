
#include "bch_bm.h"

#include <iostream>
using namespace std;

#define MAXN ((1<<16)-1)  // primitive code length

int main()
{
	char codeword[MAXN],
		message[MAXN],
		messageRec[MAXN]; // information bits
	unsigned long seed = 1;

	BCH_BM	bch;

	bch.setCode( CR_3_4, FRAME_NORMAL );

	bch.message_gen( bch.getN(), bch.getK(), &seed, message );

	bch.encode( message, codeword );

	bch.simulateError( codeword, 12 );

	bch.decode( messageRec, codeword );

	bool bStatus = bch.verifyResult( message, messageRec );

	cudaDeviceReset();

	if ( bStatus )
		cout << "succeed" << endl; 
	else
		cout << "fail" << endl; 

}