
#include "dvb2_ldpc_encode.h"
#include "dvbUtility.h"

#include <iostream>
using namespace std;

#define MAXN ((1<<16)-1)  // primitive code length
#define		FRAME_CACHE_COUNT	4

int main()
{
	Bit *codeword = new Bit[MAXN*FRAME_CACHE_COUNT];
	Bit *message = new Bit[MAXN*FRAME_CACHE_COUNT];
	Bit *messageRec = new Bit[MAXN*FRAME_CACHE_COUNT]; // information bits
	unsigned long seed = 1;

	DVB2FrameFormat	dvbs2_fmt;

	dvbs2_fmt.frame_type    = FRAME_NORMAL;
	dvbs2_fmt.code_rate     = CR_3_4;
	dvbs2_fmt.constellation = M_32APSK;
	dvbs2_fmt.roll_off      = RO_0_35;
	dvbs2_fmt.pilots        = 0;
	dvbs2_fmt.dummy_frame   = 0;
	dvbs2_fmt.null_deletion = 0;

	dvbs2_fmt.configure();

	Ldpc_encode	m_ldpc_encode;
	m_ldpc_encode.ldpc_lookup_generate( &dvbs2_fmt );

	message_gen( dvbs2_fmt.nldpc, dvbs2_fmt.kldpc, &seed, message );

	m_ldpc_encode.ldpc_encode( codeword );

	delete[]	codeword;
	delete[]	message;
	delete[]	messageRec;
}