
#include "DVBS2-decode.h"
#include "dvbUtility.h"
#include "helper_timer.h"
#define		TIME_STEP		6	

int DVBS2_DECODE::s2_decode_ts_frame( scmplx* pl )
{
	vec			timerStepValue(TIME_STEP);

	int nTimeStep = 0;
	StopWatchInterface	*timerStep;

	sdkCreateTimer( &timerStep );
	sdkStartTimer( &timerStep );

	memcpy_s( this->m_pl, sizeof(scmplx)*FRAME_SIZE_NORMAL, 
		pl, sizeof(scmplx)*FRAME_SIZE_NORMAL);

	int res = 0;
	
	// decode the header
	s2_pl_header_decode();

	set_configure();

	// Now apply the scrambler to the data part not the header
	pl_scramble_decode( &m_pl[90], m_payload_symbols );

	sdkStopTimer( &timerStep );
	timerStepValue[nTimeStep++] = sdkGetTimerValue( &timerStep );// 1.2 ms

	sdkResetTimer( &timerStep );
	sdkStartTimer( &timerStep );

	// decode the data
	if( !m_bDecodeSoft )
	{
		res = s2_demodulate_hard();

		sdkStopTimer( &timerStep );
		timerStepValue[nTimeStep++] = sdkGetTimerValue( &timerStep );// 1.6 ms, 16(d)

		sdkResetTimer( &timerStep );
		sdkStartTimer( &timerStep );

		// de-Interleave and pack	
		if( m_bInterleave )
			s2_deinterleave();
		else
			s2_i2b();

		sdkStopTimer( &timerStep );
		timerStepValue[nTimeStep++] = sdkGetTimerValue( &timerStep );// 0.2 ms
	}
	else
	{
		demodulate_soft_bits( &m_pl[90], N0, m_soft_bits_cache );

		sdkStopTimer( &timerStep );
		timerStepValue[nTimeStep++] = sdkGetTimerValue( &timerStep );// 12 ms, 490(d)

		sdkResetTimer( &timerStep );
		sdkStartTimer( &timerStep );

		if( m_bInterleave )
			reorder_softbit();
		else
			memcpy_s( m_soft_bits, sizeof(double)*FRAME_SIZE_NORMAL,
			m_soft_bits_cache, sizeof(double)*FRAME_SIZE_NORMAL );

		sdkStopTimer( &timerStep );
		timerStepValue[nTimeStep++] = sdkGetTimerValue( &timerStep );// 0.2 ms

	}
	sdkResetTimer( &timerStep );
	sdkStartTimer( &timerStep );

	ldpc_decode();// 3.8 ms, 20(d)	//cpp 78(d) 

	sdkStopTimer( &timerStep );
	timerStepValue[nTimeStep++] = sdkGetTimerValue( &timerStep );

	sdkResetTimer( &timerStep );
	sdkStartTimer( &timerStep );

	// BCH encode the BB Frame
	bch_decode();

	sdkStopTimer( &timerStep );
	timerStepValue[nTimeStep++] = sdkGetTimerValue( &timerStep );// 10 ms, 16(d)

	sdkResetTimer( &timerStep );
	sdkStartTimer( &timerStep );

	// Yes so now Scramble the BB frame
	bb_randomise_decode();

	// New frame needs to be sent
	decode_bbheader(); // Add the header

	m_frame_offset_bits += 8; // crc

	// Add a new transport packet
	transport_packet_decode_crc( m_frame );

	sdkStopTimer( &timerStep );
	timerStepValue[nTimeStep++] = sdkGetTimerValue( &timerStep );// 0.3ms

	sdkDeleteTimer( &timerStep );

	m_nTotalFrame++;
#if 1
	for (int i=0;i<TIME_STEP;i++)
	{
		cout  << "timerStepValue[ " << i << " ] = "<< timerStepValue[i] << " ms, " << endl;
	}
#endif
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
	{
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
	}

	if ( modcod >= 12 && modcod <= 17 )
	{
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
	}

	if ( modcod >= 18 && modcod <= 23 )
	{
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
	}

	if ( modcod >= 24 && modcod <= 28 )
	{
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

	// bit[5]
	bit = 0x80000000;
	if( temp&bit )
	{
		in ^= 0x02;
		temp ^= g[5];
	}

	// bit[0]
	bit = 0x40000000;
	if( temp&bit )
	{
		in ^= 0x40;
		temp ^= g[0];
	}

	// bit[1]
	bit = 0x20000000;
	if( temp&bit )
	{
		in ^= 0x20;
		temp ^= g[1];
	}

	// bit[2]
	bit = 0x08000000;
	if( temp&bit )
	{
		in ^= 0x10;
		temp ^= g[2];
	}	

	// bit[3]
	bit = 0x00800000;
	if( temp&bit )
	{
		in ^= 0x08;
		temp ^= g[3];
	}

	// bit[4]
	bit = 0x00008000;
	if( temp&bit )
	{
		in ^= 0x04;
		temp ^= g[4];
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

int DVBS2_DECODE::s2_demodulate_hard()
{

	int m = 0;
	int n = 90;// Jump over header
	int blocks = m_payload_symbols/90;
	//int block_count = 0;
	
	for( int i = 0; i < blocks; i++ )
	{
		for( int j = 0; j < 90; j++ )
		{
			m_iframe[m++] = demodulate_hard(m_pl[n++]);//m_pl[n++] = m_qpsk[m_iframe[m++]];
		}
		// Add pilots if needed
		// ... todo 
	}


	// Return the length
	return n;
}

void DVBS2_DECODE::s2_deinterleave()
{
	int rows=0;

	int frame_size = m_format[0].nldpc;

	// no interleave
	if( m_format[0].constellation == M_QPSK )
	{
		rows = frame_size/2;
		Bit *c1,*c2;

		for( int i = 0; i < rows; i++ )
		{
			m_frame[i*2]	= (m_iframe[i]>>1) & 1;
			m_frame[i*2+1]	= m_iframe[i] & 1;

		}
		return;
	}

	// interleave
	int nConstellationType = m_format[0].constellation + 2;
	rows = frame_size / nConstellationType;

	for( int i = 0; i < rows; i++ )
		for (int j=0;j<nConstellationType;j++)
			if( m_format[0].constellation == M_8PSK && 
				m_format[0].code_rate == CR_3_5 )
				m_frame[j*rows+i] = m_iframe[i]>>j & 1;			// MSB of BBHeader first
			else
				m_frame[j*rows+i] = m_iframe[i]>>(nConstellationType-1-j) & 1;	// third
	
	return;
}
#if 0
bool DVBS2_DECODE::decode_ts_frame_base( Bit* b )
{
	if( m_frame_offset_bits == 0 )
	{
		// LDPC encode the BB frame and BCHFEC bits
		ldpc_decode();

		// BCH encode the BB Frame
		bch_decode();

		// Yes so now Scramble the BB frame
		bb_randomise_decode();

		// New frame needs to be sent
		decode_bbheader(); // Add the header

		m_frame_offset_bits += 8; // crc
	}

	memset( msg, 0, sizeof(u8)*PACKET_SIZE );

	// Add a new transport packet
	transport_packet_decode_crc( b );

	return 1;
}
#endif

void DVBS2_DECODE::ldpc_decode()
{
	// b m_frame[N] -> b m_frame[K]
	if ( !m_bDecodeSoft )
		return;

	if( !m_bUseGPU )
		ldpc.bp_decode( m_soft_bits, m_bitLDPC, m_format[0].code_rate  );
#if	USE_GPU
	else
		ldpc_gpu.bp_decode_once( m_soft_bits, m_bitLDPC, m_format[0].code_rate );
#endif

	// interleave
	int rows=0;

	int frame_size = m_format[0].nldpc;

	int nConstellationType = m_format[0].constellation + 2;
	rows = frame_size / nConstellationType;

	for( int i = 0; i < rows; i++ )
		for (int j=0;j<nConstellationType;j++)
				m_frame[i*nConstellationType+j] = m_bitLDPC[i*nConstellationType+j];
}

void DVBS2_DECODE::bch_decode()
{
	// b m_frame[n] -> b m_frame[k] 
	if ( !m_bDecodeSoft )
		return;

	for(int i=0;i<FRAME_SIZE_NORMAL;i++)
		m_bitLDPC[i] = m_frame[i];

	bch.decode( m_bitBCH, m_bitLDPC );

	for(int i=0;i<FRAME_SIZE_NORMAL;i++)
		m_frame[i] = m_bitBCH[i];

}

void DVBS2_DECODE::bb_randomise_decode()
{
	for( int i = 0; i < m_format[0].kbch; i++ )
	{
		m_frame[i] ^= m_bb_randomise[i];
	}
}

void DVBS2_DECODE::transport_packet_decode_crc( Bit* b )
{
	memset( msg, 0, sizeof(u8)*FRAME_SIZE_NORMAL/8 );

	int nByteCount = m_format[0].bb_header.dfl/8;
	for( int i = 0; i < nByteCount; i++ )
	{	
		for( int n = 7; n >= 0; n-- )
		{
			msg[i] += m_frame[m_frame_offset_bits++] << n;
		}
	}
}

bool DVBS2_DECODE::decode_bbheader()
{
	bool bStatus = true;

	int temp;

	BBHeader *h = &m_format[0].bb_header;

	// First byte (MATYPE-1)
	h->ts_gs	=  m_frame[0] << 1;
	h->ts_gs	+= m_frame[1];
	h->sis_mis	=  m_frame[2];
	h->ccm_acm	=  m_frame[3];
	h->issyi	=  m_frame[4];
	h->npd		=  m_frame[5];

	h->ro		= m_frame[6] << 1;
	h->ro		+= m_frame[7];

	m_frame_offset_bits = 8;

	// Second byte (MATYPE-2)
	if (h->sis_mis == SIS_MIS_MULTIPLE)
	{
		temp = 0;
		for (int n = 7; n >= 0; n--)
		{
			temp += m_frame[m_frame_offset_bits++] << n;// = temp & (1 << n) ? 1 : 0;
		}
		h->isi = temp;
	}
	else
	{
		for (int n = 7; n >= 0 ; n--)
		{
			m_frame_offset_bits++;
		}
	}

	// UPL (2 bytes)
	temp = 0;
	for (int n = 15; n >= 0; n--)
	{
		temp += m_frame[m_frame_offset_bits++] << n;
	}
	h->upl = temp;

	// DFL (2 byte)
	temp = 0;
	for (int n = 15; n >= 0; n--)
	{
		temp += m_frame[m_frame_offset_bits++] << n;
	}
	h->dfl = temp;

	// SYNC (1 byte)
	temp = 0;
	for (int n = 7; n >= 0; n--)
	{
		temp += m_frame[m_frame_offset_bits++] << n;
	}
	h->sync = temp;

	// Calculate syncd (2 byte), this should point to the MSB of the CRC
	temp = 0;
	for (int n = 15; n >= 0; n--)
	{
		temp += m_frame[m_frame_offset_bits++] << n;
	}
	h->syncd = temp;

	return bStatus;
}

DVBS2_DECODE::DVBS2_DECODE()
{
	m_bDecodeSoft = true;

	m_bUseGPU = true;
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

int DVBS2_DECODE::demodulate_hard( const scmplx& sym )
{
	// m_payload_symbols
	scmplx*	pSymbolsTemplate = NULL;
	int nSymbolSize = -1;
	switch( m_format[0].constellation )
	{
	case M_QPSK:
		pSymbolsTemplate = m_qpsk;
		nSymbolSize = 1<<2;
		break;
	case M_8PSK:
		pSymbolsTemplate = m_8psk;
		nSymbolSize = 1<<3;
		break;
	case M_16APSK:
		pSymbolsTemplate = m_16apsk;
		nSymbolSize = 1<<4;
		break;
	case M_32APSK:
		pSymbolsTemplate = m_32apsk;
		nSymbolSize = 1<<5;
		break;
	default:
		break;
	}

	int index = demodulate_hard( sym, pSymbolsTemplate, nSymbolSize );

	return index;
}

int DVBS2_DECODE::demodulate_hard( const scmplx& sym, scmplx* const symTemplate, int n )
{
	int closest = -1;
	float mindist = 0, dist = 0;

	mindist = distance(sym, symTemplate[0]);
	closest = 0;

	for (int j = 1; j < n; j++) {
		dist = distance(sym, symTemplate[j]);
		if (dist < mindist) {
			mindist = dist;
			closest = j;
		}
	}

	return closest;
}

float DVBS2_DECODE::distance( const scmplx& cL, const scmplx& cR )
{
	float dist2 = 0;
	dist2 = (float)(cL.im - cR.im) * (cL.im - cR.im) + (float)(cL.re - cR.re) * (cL.re - cR.re);
	return dist2;
}

void DVBS2_DECODE::pl_scramble_decode( scmplx *fs, int len )
{
	scmplx x;

	// Start at the end of the PL Header.

	for( int n = 0; n < len; n++ )
	{
		x = fs[n];
		switch( m_cscram[n] )
		{
		case 0:
			// Do nothing
			break;
		case 1:
			fs[n].re =  x.im;
			fs[n].im = -x.re;
			break;
		case 2:
			fs[n].re = -fs[n].re;
			fs[n].im = -fs[n].im;
			break;
		case 03:
			x = fs[n];
			fs[n].re = -x.im;
			fs[n].im =  x.re;
			break;
		}
	}
}

unsigned char* DVBS2_DECODE::getByte()
{
	return msg;
}

void DVBS2_DECODE::demodulate_soft_bits( scmplx* sym, double N0, double* soft_bits )
{

	Modulator_2D* pModulator = mods.findModulator( m_format[0].constellation );

	cvec	cAWGN( m_payload_symbols );

	for (int i = 0; i< cAWGN.size(); i++ ){
		cAWGN._elem(i).real( sym[i].re * 1.0/CP );
		cAWGN._elem(i).imag( sym[i].im * 1.0/CP );
	}

	vec softbits = pModulator->demodulate_soft_bits(cAWGN, N0,APPROX);

	convertVecToBuffer( soft_bits, softbits );
}

float DVBS2_DECODE::get_rate()
{
	float rate = m_format[0].kldpc * 1.0f / m_format[0].nldpc;

	return rate;
}

void DVBS2_DECODE::set_configure()
{
	// config dvb-s2 format 
	s2_set_configure( &m_format[0] );
	N0 = pow(10.0, -EBNO / 10.0) / get_rate();

	// m_payload_symbols
	int frame_size = m_format[0].nldpc;
	switch( m_format[0].constellation )
	{
	case M_QPSK:
		m_payload_symbols = frame_size / 2;
		break;
	case M_8PSK:
		m_payload_symbols = frame_size / 3;
		break;
	case M_16APSK:
		m_payload_symbols = frame_size / 4;
		break;
	case M_32APSK:
		m_payload_symbols = frame_size / 5;
		break;
	default:
		break;
	}

	bch.setCode( m_format[0].code_rate, m_format[0].frame_type );

}

void DVBS2_DECODE::initialize()
{
	m_codes.initialize();

	ldpc.initialize(&m_codes);

	bch.initialize();
#if USE_GPU
	ldpc_gpu.initialize(&m_codes);
#endif
}

void DVBS2_DECODE::s2_i2b()
{
	int rows=0;

	int frame_size = m_format[0].nldpc;

	// interleave
	int nConstellationType = m_format[0].constellation + 2;
	rows = frame_size / nConstellationType;

	for( int i = 0; i < rows; i++ )
		for (int j=0;j<nConstellationType;j++)
			m_frame[i*nConstellationType+j] = m_iframe[i]>>(nConstellationType-1-j) & 1;	// third

	return;
}

void DVBS2_DECODE::reorder_softbit()
{
	int rows=0;

	int frame_size = m_format[0].nldpc;

	// interleave
	int nConstellationType = m_format[0].constellation + 2;
	rows = frame_size / nConstellationType;

	for (int j=0;j<nConstellationType;j++)
		for( int i = 0; i < rows; i++ )
			m_soft_bits[j*rows+i] = m_soft_bits_cache[i*nConstellationType+j];	

	return;
}
