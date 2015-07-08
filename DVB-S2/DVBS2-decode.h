
#ifndef DVBS2_DECODE_H
#define DVBS2_DECODE_H

#include "DVBS2.h"
#include "itpp/itcomm.h"
#include "ldpc_bp_decode.h"
#include "modulatorFactory.h"
#include "bch_bm.h"

#ifdef USE_GPU
#include "ldpc_bp_decode.cuh"
#endif

#define PACKET_SIZE		188

#define		EBNO			2.6//10 2-2.2	3-5.6	4-8.9	5-12.4

class DVBS2_DECODE : public DVBS2
{
public:
	DVBS2_DECODE();
	~DVBS2_DECODE();

	int s2_decode_ts_frame( scmplx* pl );	// c m_pl[]	->	B ts[]

	unsigned char* getByte();

	void initialize();

protected:
	void s2_pl_header_decode();	// c m_pl[90]	->	i MODCOD
	int s2_demodulate_hard();	// c m_pl[>90]	->	i m_iframe
	void s2_deinterleave();		// i m_iframe	->	b m_frame 
	void s2_i2b();				// i m_iframe	->	b m_frame 
	void reorder_softbit();

	void ldpc_decode();
	void bch_decode();
	void bb_randomise_decode();
	void transport_packet_decode_crc( Bit* b );
	bool decode_bbheader();

	int	checkSOF(int* sof, int n);
	void s2_pl_header_decode( u8* modcod, u8* type, int *b);
	void b_64_7_decode( unsigned char *c, int *b );

	int demodulate_hard( const scmplx& sym );
	int demodulate_hard( const scmplx& sym, scmplx* const symTemplate, int n );
	float distance( const scmplx& cL, const scmplx& cR );

	void pl_scramble_decode( scmplx *fs, int len );

	void set_configure();
	void demodulate_soft_bits( scmplx* sym, double N0, double* soft_bits );
	float get_rate();
	void decode_soft( scmplx* sym, double N0 );
	void configFormatByTypeModcod( u8 type, u8 modcod ); 

private:
	u8	msg[FRAME_SIZE_NORMAL/8];
	double N0;
	double	m_soft_bits[FRAME_SIZE_NORMAL];
	double	m_soft_bits_cache[FRAME_SIZE_NORMAL];
	char	m_bitLDPC[FRAME_SIZE_NORMAL];
	char	m_bitBCH[FRAME_SIZE_NORMAL];

	ldpc_decoder	ldpc;
	bool	m_bDecodeSoft;

	ModulatorFactory	mods;	// 调制解调器件库
	BCH_BM	bch;

#ifdef USE_GPU
	ldpc_gpu	m_ldpc_gpu;
#endif
	bool	m_bUseGPU;

	LDPC_CodeFactory	m_codes;

	scmplx*	pSymbolsTemplate ;
	int nSymbolSize ;

	scmplx	m_Symbols[M_CONST_NUMBER][32] ;

	u8		m_typeLast, m_modcodLast;
	bool	m_bNeedUpdateCode;
};

#endif