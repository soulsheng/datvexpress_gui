
#include "DVBS2.h"

#define PACKET_NUMBER	100
#define PACKET_SIZE		188

void main()
{
	DVBS2*	m_dvbs2 = new DVBS2;

	u8 b[PACKET_NUMBER*PACKET_SIZE];

	m_dvbs2->s2_add_ts_frame( b );

	delete	m_dvbs2;
}