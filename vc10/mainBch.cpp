
#include "bch_bm.h"
#include "dvbUtility.h"
#include <iostream>
using namespace std;

#define MAXN ((1<<16)-1)  // primitive code length
#define		FRAME_CACHE_COUNT	4

int main()
{
	char codeword[MAXN*FRAME_CACHE_COUNT],
		message[MAXN*FRAME_CACHE_COUNT],
		messageRec[MAXN*FRAME_CACHE_COUNT]; // information bits
	unsigned long seed = 1;

	BCH_BM	bch;

	bch.initialize( FRAME_CACHE_COUNT );
	bch.setCode( CR_3_4, FRAME_NORMAL );

	message_gen( bch.getN(), bch.getK(), &seed, message );

	bch.encode( message, codeword );

	bch.simulateError( codeword, 12 );

	for(int i = 0;i<FRAME_CACHE_COUNT;i++)
	memcpy_s( codeword + bch.getN()*i, bch.getN() * sizeof(char), codeword, bch.getN() * sizeof(char) );

	bch.decode( messageRec, codeword , FRAME_CACHE_COUNT );

	bool bStatus = bch.verifyResult( messageRec, message, FRAME_CACHE_COUNT );

	cudaDeviceReset();

	if ( bStatus )
		cout << "succeed" << endl; 
	else
		cout << "fail" << endl; 

}