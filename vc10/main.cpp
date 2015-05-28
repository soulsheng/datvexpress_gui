
#include "DVBS2.h"
#include "DVBS2-decode.h"

#define PACKET_NUMBER	100
#define PACKET_STREAM	(PACKET_NUMBER*PACKET_SIZE)
#define CP 0x7FFF
#define PRINT_SIZE		16

void init(u8* buffer, int n);	// initialize info
void print(scmplx* c, int n);	// output encoded info
void print(u8* b, int n);		// output original info

void main()
{
	DVBS2*	m_dvbs2 = new DVBS2;
	DVBS2_DECODE*	m_dvbs2_dec = new DVBS2_DECODE;
	DVB2FrameFormat	dvbs2_fmt;
	//
	// DVB-S2
	//
	dvbs2_fmt.frame_type    = FRAME_NORMAL;
	dvbs2_fmt.code_rate     = CR_3_4;
	dvbs2_fmt.constellation = M_32APSK;
	dvbs2_fmt.roll_off      = RO_0_35;
	dvbs2_fmt.pilots        = 0;
	dvbs2_fmt.dummy_frame   = 0;
	dvbs2_fmt.null_deletion = 0;
	int nStatus = m_dvbs2->s2_set_configure( &dvbs2_fmt );
	if( -1 == nStatus )
	{
		printf(" mode(%d, %d) is invalid ! \n",
			dvbs2_fmt.constellation, dvbs2_fmt.code_rate );

		delete	m_dvbs2;
		delete	m_dvbs2_dec;

		return ;
	}

	u8 b[PACKET_STREAM], bRef[PACKET_STREAM];
	init( b, PACKET_STREAM );
	print( b, PACKET_STREAM );

	for (int i=0;i<PACKET_NUMBER;i++)
		m_dvbs2->s2_add_ts_frame( b + i*PACKET_SIZE );

	scmplx* c = m_dvbs2->pl_get_frame();
	int		nSymbol = m_dvbs2->s2_get_n_symbol();
	print( c, nSymbol );

	m_dvbs2_dec->s2_decode_ts_frame( c );
	print( m_dvbs2_dec->getByte(), PACKET_SIZE );

	delete	m_dvbs2;
	delete	m_dvbs2_dec;
}

void init(u8* buffer, int n)	// initialize info
{
	for (int i=0;i<n;i++)
		buffer[i] = i%256;
}

void print(scmplx* c, int n)	// output encoded info
{
	int nPrint = PRINT_SIZE;//n;
	for (int i=0;i<nPrint;i++)
		printf("%d: (%hd,%hd), (%f,%f)\n", i, c[i].re, c[i].im,
		c[i].re*1.0f/CP, c[i].im*1.0f/CP );
}

void print(u8* b, int n)		// output original info
{
	int nPrint = PRINT_SIZE;//n;
	for (int i=0;i<nPrint;i++)
		printf("%d: %d \n", i, b[i] );
}