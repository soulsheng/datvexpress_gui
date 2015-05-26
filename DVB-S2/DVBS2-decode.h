
#ifndef DVBS2_DECODE_H
#define DVBS2_DECODE_H

#include "DVBS2.h"

#define PACKET_SIZE		188

class DVBS2_DECODE : public DVBS2
{
public:
	DVBS2_DECODE();
	~DVBS2_DECODE();

	int s2_decode_ts_frame( scmplx* pl );	// c m_pl[]	->	B ts[]
	void s2_pl_header_decode();	// c m_pl[90]	->	i MODCOD
	int s2_pl_data_decode();	// c m_pl[>90]	->	i m_iframe
	void s2_deinterleave();		// i m_iframe	->	b m_frame 

	bool decode_ts_frame_base( Bit* b );
	void ldpc_decode();
	void bch_decode();
	void bb_randomise_decode();
	void transport_packet_decode_crc( Bit* b );
	bool decode_bbheader();
	unsigned char* getByte();

protected:
	int	checkSOF(int* sof, int n);
	void s2_pl_header_decode( u8* modcod, u8* type, int *b);
	void b_64_7_decode( unsigned char *c, int *b );

	int demodulate_hard( const scmplx& sym );
	int demodulate_hard( const scmplx& sym, scmplx* const symTemplate, int n );
	float distance( const scmplx& cL, const scmplx& cR );

	void pl_scramble_decode( scmplx *fs, int len );

private:
	u8	msg[FRAME_SIZE_NORMAL/8];
};

#endif