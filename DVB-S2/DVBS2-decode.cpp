
#include "DVBS2-decode.h"

int DVBS2_DECODE::s2_decode_ts_frame( scmplx* pl )
{
	memcpy_s( pl_get_frame(), sizeof(scmplx)*FRAME_SIZE_NORMAL, 
		pl, sizeof(scmplx)*FRAME_SIZE_NORMAL);

	int res = 0;
	
	// decode the header
	s2_pl_header_decode();
	// decode the data
	res = s2_pl_data_decode();
	// de-Interleave and pack
	s2_deinterleave();
		
	while( decode_ts_frame_base( m_frame ) );

	return res;
}

void DVBS2_DECODE::s2_pl_header_decode()
{
	int b[90];

	// BPSK modulate and add the header
	for( int i = 0; i < 90; i++ )
	{
		if (	m_pl[i].im == m_bpsk[i&1][0].im
			&&	m_pl[i].re == m_bpsk[i&1][0].re )
			b[i] = 0;
		else
			b[i] = 1;
	}

	// check the sync sequence SOF
	int nStat = checkSOF( b, 26 );
	if( -1 != nStat )
		printf(" sof[%d] error \n ", nStat );

	u8 type, modcod;
	// check the mode and code
	s2_pl_header_decode( &modcod, &type, &b[26] );


	if( type&0x02 )
		m_format[0].frame_type = FRAME_SHORT;
	else
		m_format[0].frame_type = FRAME_NORMAL;

	if( type&0x01 ) 
		m_format[0].pilots = 1;
	else
		m_format[0].pilots = 0;


	// Mode and code rate

	if ( modcod <= 11 )
		m_format[0].constellation = M_QPSK;

	switch( modcod )
	{
	case 1:
		m_format[0].code_rate = CR_1_4;
		break;
	case 2:
		m_format[0].code_rate = CR_1_3;
		break;
	case 3:
		m_format[0].code_rate = CR_2_5;
		break;
	case 4:
		m_format[0].code_rate = CR_1_2;
		break;
	case 5:
		m_format[0].code_rate = CR_3_5;
		break;
	case 6:
		m_format[0].code_rate = CR_2_3;
		break;
	case 7:
		m_format[0].code_rate = CR_3_4;
		break;
	case 8:
		m_format[0].code_rate = CR_4_5;
		break;
	case 9:
		m_format[0].code_rate = CR_5_6;
		break;
	case 10:
		m_format[0].code_rate = CR_8_9;
		break;
	case 11:
		m_format[0].code_rate = CR_9_10;
		break;
	default:
		break;
	}

	if ( modcod >= 12 && modcod <= 17 )
		m_format[0].constellation = M_8PSK;


	switch( modcod )
	{
	case 12:
		m_format[0].code_rate = CR_3_5;
		break;
	case 13:
		m_format[0].code_rate = CR_2_3;
		break;
	case 14:
		m_format[0].code_rate = CR_3_4;
		break;
	case 15:
		m_format[0].code_rate = CR_5_6;
		break;
	case 16:
		m_format[0].code_rate = CR_8_9;
		break;
	case 17:
		m_format[0].code_rate = CR_9_10;
		break;
	default:
		break;
	}

	if ( modcod >= 18 && modcod <= 23 )
		m_format[0].constellation = M_16APSK;


	switch( modcod )
	{
	case 18:
		m_format[0].code_rate = CR_2_3;
		break;
	case 19:
		m_format[0].code_rate = CR_3_4;
		break;
	case 20:
		m_format[0].code_rate = CR_4_5;
		break;
	case 21:
		m_format[0].code_rate = CR_5_6;
		break;
	case 22:
		m_format[0].code_rate = CR_8_9;
		break;
	case 23:
		m_format[0].code_rate = CR_9_10;
		break;
	default:
		break;
	}


	if ( modcod >= 24 && modcod <= 28 )
		m_format[0].constellation = M_32APSK;


	switch( modcod )
	{
	case 24:
		m_format[0].code_rate = CR_3_4;
		break;
	case 25:
		m_format[0].code_rate = CR_4_5;
		break;
	case 26:
		m_format[0].code_rate = CR_5_6;
		break;
	case 27:
		m_format[0].code_rate = CR_8_9;
		break;
	case 28:
		m_format[0].code_rate = CR_9_10;
		break;
	default:
		break;
	}
	return;
}

void DVBS2_DECODE::b_64_7_decode( unsigned char *c, int *b )
{
	unsigned char &in = *c;
	int* out = b;
	unsigned long temp,bit;

	// Randomise it
	for( int m = 0; m < 64; m++ )
	{
		out[m] = out[m] ^ ph_scram_tab[m];
	}

	temp = 0;

	in = out[0] ^ out[1];
	bit = 0x80000000;
	for( int m = 0; m < 32; m++ )
	{
		if ( out[(m*2)] )
			temp |= bit;
		bit >>= 1;
	}

	bit = 0x80000000;
	if( temp&bit )
	{
		in ^= 0x02;
		temp ^= g[5];
	}


	bit = 0x00001000;
	if( temp&bit )
	{
		in ^= 0x04;
		temp ^= g[4];
	}

	bit = 0x00100000;
	if( temp&bit )
	{
		in ^= 0x08;
		temp ^= g[3];
	}

	bit = 0x08000000;
	if( temp&bit )
	{
		in ^= 0x10;
		temp ^= g[2];
	}

	bit = 0x20000000;
	if( temp&bit )
	{
		in ^= 0x20;
		temp ^= g[1];
	}

	bit = 0x40000000;
	if( temp&bit )
	{
		in ^= 0x40;
		temp ^= g[0];
	}

}

void DVBS2_DECODE::s2_pl_header_decode( u8* modcod, u8* type, int *b )
{
	unsigned char code;

	//printf("MODCOD %d TYPE %d %d\n",modcod,type,code);
	// Add the modcod and type information and scramble it
	b_64_7_decode( &code, b );

	*type = code & 3;
	*modcod = code >>2;
}

int DVBS2_DECODE::s2_pl_data_decode()
{
	int n = 90;// Jump over header

	return n;
}

void DVBS2_DECODE::s2_deinterleave()
{

}

bool DVBS2_DECODE::decode_ts_frame_base( Bit* b )
{
	if( m_frame_offset_bits == 0 )
	{
		// New frame needs to be sent
		decode_bbheader(); // Add the header

		// Yes so now Scramble the BB frame
		bb_randomise_decode();
		// BCH encode the BB Frame
		bch_decode();
		// LDPC encode the BB frame and BCHFEC bits
		ldpc_decode();
		return 1;
	}

	memset( msg, 0, sizeof(u8)*PACKET_SIZE );
	// Add a new transport packet
	while( m_frame_offset_bits != m_format[0].kbch )
		transport_packet_decode_crc( b );

	return 0;
}

void DVBS2_DECODE::ldpc_decode()
{

}

void DVBS2_DECODE::bch_decode()
{

}

void DVBS2_DECODE::bb_randomise_decode()
{

}

void DVBS2_DECODE::transport_packet_decode_crc( Bit* b )
{
	for( int i = 0; i < PACKET_SIZE; i++ )
	{	
		for( int n = 7; n >= 0; n-- )
		{
			msg[i] += m_frame[m_frame_offset_bits++] << n;
		}
	}
}

void DVBS2_DECODE::decode_bbheader()
{

}

DVBS2_DECODE::DVBS2_DECODE()
{

}

DVBS2_DECODE::~DVBS2_DECODE()
{

}

int DVBS2_DECODE::checkSOF( int* sof, int n )
{
	for( int i = 0; i < 26; i++ ) 
	{
		if( sof[i] != ph_sync_seq[i] )
			return i;
	}
	return -1;
}
