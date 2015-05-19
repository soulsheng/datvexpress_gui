
#include "DVBS2.h"

#define PACKET_NUMBER	100
#define PACKET_SIZE		188

void main()
{
	DVBS2*	m_dvbs2 = new DVBS2;
	DVB2FrameFormat	dvbs2_fmt;
	//
	// DVB-S2
	//
	dvbs2_fmt.frame_type    = FRAME_NORMAL;
	dvbs2_fmt.code_rate     = CR_1_2;
	dvbs2_fmt.constellation = M_QPSK;
	dvbs2_fmt.roll_off      = RO_0_35;
	dvbs2_fmt.pilots        = 0;
	dvbs2_fmt.dummy_frame   = 0;
	dvbs2_fmt.null_deletion = 0;
	m_dvbs2->s2_set_configure( &dvbs2_fmt );

	u8 b[PACKET_NUMBER*PACKET_SIZE];

	for (int i=0;i<PACKET_NUMBER;i++)
		m_dvbs2->s2_add_ts_frame( b + i*PACKET_SIZE );

	delete	m_dvbs2;
}