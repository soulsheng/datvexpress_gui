#include "memory.h"
#include "DVBS2.h"

//
// called at the end of a frame
//
void  DVBS2::end_of_frame_actions(void)
{
    if( m_s2_config_updated )
    {
        modulator_configuration();
        m_s2_config_updated = 0;
    }

	memcpy_s( m_pl_cache+m_nTotalFrame*FRAME_SIZE_NORMAL, sizeof(scmplx)*FRAME_SIZE_NORMAL,
		m_pl, sizeof(scmplx)*FRAME_SIZE_NORMAL );

	m_nTotalFrame ++ ;
}

int DVBS2::is_valid( int mod, int coderate )
{
    if( mod == M_QPSK )
    {
        if(coderate == CR_1_4)  return 0;
        if(coderate == CR_1_3)  return 0;
        if(coderate == CR_2_5)  return 0;
        if(coderate == CR_1_2)  return 0;
        if(coderate == CR_3_5)  return 0;
        if(coderate == CR_2_3)  return 0;
        if(coderate == CR_3_4)  return 0;
        if(coderate == CR_4_5)  return 0;
        if(coderate == CR_5_6)  return 0;
        if(coderate == CR_8_9)  return 0;
        if(coderate == CR_9_10) return 0;
    }
    if( mod == M_8PSK )
    {
        if(coderate == CR_3_5)  return 0;
        if(coderate == CR_2_3)  return 0;
        if(coderate == CR_3_4)  return 0;
        if(coderate == CR_5_6)  return 0;
        if(coderate == CR_8_9)  return 0;
        if(coderate == CR_9_10) return 0;
    }
    if( mod == M_16APSK )
    {
        if(coderate == CR_2_3)  return 0;
        if(coderate == CR_3_4)  return 0;
        if(coderate == CR_4_5)  return 0;
        if(coderate == CR_5_6)  return 0;
        if(coderate == CR_8_9)  return 0;
        if(coderate == CR_9_10) return 0;
    }
    if( mod == M_32APSK )
    {
        if(coderate == CR_3_4)  return 0;
        if(coderate == CR_4_5)  return 0;
        if(coderate == CR_5_6)  return 0;
        if(coderate == CR_8_9)  return 0;
        if(coderate == CR_9_10) return 0;
    }
    return -1;
}
//
// index 0 and 1 will only be different when being reconfigured.
// Use index 1 as this will be applied in the following transmit
// frames
//
void DVBS2::calc_efficiency( void )
{
    double p,m,a,s,b,po;
    // Calculate the number of symbols in the payload
    p = 0;a = 0; m = 0;
    if( m_format[1].frame_type == FRAME_NORMAL )  p = (double)FRAME_SIZE_NORMAL;
    if( m_format[1].frame_type == FRAME_SHORT  )  p = (double)FRAME_SIZE_SHORT;
    if( m_format[1].constellation == M_QPSK )     m = 2.0;
    if( m_format[1].constellation == M_8PSK )     m = 3.0;
    if( m_format[1].constellation == M_16APSK )   m = 4.0;
    if( m_format[1].constellation == M_32APSK )   m = 5.0;
    s = p/m;//Number of symbols per frame
    // PL header overhead
    if( m_format[1].pilots )
    {
        po = (s/(90*16))-1;// 1 pilot every 16 blocks (of 90 symbols)
        po = po*36;        // No pilot at the end
        a  = s/(90+po+s);
    }
    else
    {
        a = s/(90+s);// No pilots
    }
    // Modulation efficiency
    a = a*m;
    // Take into account pilot symbols
    // TBD
    // Now calculate the useable data as percentage of the frame
    b = ((double)m_format[1].useable_data_bits)/p;
    // Now calculate the efficiency by multiplying the
    // useable bits efficiency by the modulation efficiency
    m_efficiency = b*a;
}
//
// Multiply the efficiency value by the symbol rate
// to get the useable bitrate
//
double DVBS2::s2_get_efficiency( void )
{
    return m_efficiency;
}

int DVBS2::s2_set_configure( DVB2FrameFormat *f )
{
    if( is_valid( f->constellation, f->code_rate ) == 0 )
    {
        if( set_configure( f ) == 0 )
        {
			 modulator_configuration();
             calc_efficiency();
             m_s2_config_updated = 1;
			 m_interleave.initialize( f );
             return 0;
        }
    }
    return -1;
}
void DVBS2::s2_get_configure( DVB2FrameFormat *f )
{
    get_configure( f );
}
int DVBS2::s2_add_ts_frame( u8 *ts )
{
	if( m_nTotalFrame >= FRAME_CACHE_COUNT )
	{
		printf("reach cache limit %d frames\n",FRAME_CACHE_COUNT);
		return -1;
	}

    int res = 0;
    // Call base class
    if( next_ts_frame_base( ts ) )
    {
        // Interleave and pack
		if( m_bInterleave )
			m_interleave.s2_interleave( m_iframe, m_frame, m_payload_symbols );
		else
			m_interleave.s2_b2i( m_iframe, m_frame, m_payload_symbols );
        // create the header
        s2_pl_header_create();
        // Add the data
        res = s2_pl_data_pack();
        // Do any updates required for the next frame
        end_of_frame_actions();
    }
    return res;
}
void DVBS2::physical( void )
{

}

DVBS2::DVBS2()
{
    modulator_configuration();
    build_symbol_scrambler_table();
    pl_build_dummy();

	m_nTotalFrame = 0;
	m_bInterleave = true;

	m_nMulti = FRAME_CACHE_COUNT;
	m_frameMulti = new Bit[ m_nMulti * FRAME_SIZE_NORMAL];

	m_frame = m_frameMulti;

	m_pl_cache = new scmplx[ m_nMulti * FRAME_SIZE_NORMAL ];
}

int DVBS2::s2_get_n_symbol()
{
	double p,m,S;
	// Calculate the number of symbols in the payload
	p = 0;m = 0;
	if( m_format[1].frame_type == FRAME_NORMAL )  p = (double)FRAME_SIZE_NORMAL;
	if( m_format[1].frame_type == FRAME_SHORT  )  p = (double)FRAME_SIZE_SHORT;
	if( m_format[1].constellation == M_QPSK )     m = 2.0;
	if( m_format[1].constellation == M_8PSK )     m = 3.0;
	if( m_format[1].constellation == M_16APSK )   m = 4.0;
	if( m_format[1].constellation == M_32APSK )   m = 5.0;
	S= p/m/90;//Number of symbols per frame

	int nSymbol = 90*(S+1);
	int nPilot = 36 * int((S-1)/16 );
	if( m_format[1].pilots )
		nSymbol += nPilot;

	return nSymbol;
}

int DVBS2::get_frame_count()
{
	return	m_nTotalFrame;
}

DVBS2::~DVBS2()
{
	delete[]	m_frameMulti;
	m_frameMulti = NULL;
	delete[]	m_pl_cache;
	m_pl_cache = NULL;
}

