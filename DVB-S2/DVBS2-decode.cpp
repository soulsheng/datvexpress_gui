
#include "DVBS2-decode.h"

int DVBS2_DECODE::s2_decode_ts_frame( scmplx* pl )
{
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
