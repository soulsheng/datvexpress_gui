#include "memory.h"
#include "DVB2.h"
//#include "dvb.h"

//
// Update working parameters for the next frame
// This prevents parameters changing during a frame
//
void DVB2::base_end_of_frame_actions( DVB2FrameFormat *f )
{
    if( m_params_changed )
    {
        m_format[0] = m_format[1];
        m_ldpc.ldpc_lookup_generate( f );
        m_params_changed = 0;
    }
    // reset the pointer
    m_frame_offset_bits = 0;
}

//
// This configures the system and calculates
// any required intermediate values
//
int DVB2::set_configure( DVB2FrameFormat *f )
{
    int error = 0;
	
	error = f->configure();

    if( error == 0 )
    {
        // Save the configuration, will be updated on next frame
        m_format[1] = *f;
        // reset various pointers
        m_dnp   = 0;// No deleted null packets
        // Signal we need to update on the next frame.
        if( m_params_changed )
            base_end_of_frame_actions( f );
        else
            m_params_changed = 1;
    }

	bch.setCode( f->code_rate, f->frame_type );

	return error;
}
void DVB2::get_configure( DVB2FrameFormat *f )
{
    *f = m_format[1];
}

// Points to first byte in transport packet

int DVB2::add_ts_frame_base( u8 *ts )
{
    if( m_frame_offset_bits == 0 )
    {
         // New frame needs to be sent
        add_bbheader(); // Add the header
    }
    // Add a new transport packet
    unpack_transport_packet_add_crc( ts );
    // Have we reached the end?
    if( m_frame_offset_bits == m_format[0].kbch )
    {
        // Yes so now Scramble the BB frame
        bb_randomise();
        // BCH encode the BB Frame
        bch_encode();
        // LDPC encode the BB frame and BCHFEC bits
        m_ldpc.ldpc_encode( m_frame );
        // Signal to the modulation specific class we have something to send
        base_end_of_frame_actions( m_format );
        return 1;
    }
    return 0;
}
//
// Dump NULL packets appends a counter to the end of each UP
// it is not implemented at the moment.
//
int DVB2::next_ts_frame_base( u8 *ts )
{
    int res = 0;
    // See if we need to dump null packets
    if( m_format[0].null_deletion == 1 )
    {
        if(((ts[0]&0x1F) == 0x1F)&&(ts[1] == 0xFF ))
        {
            // Null packet detected
            if( m_dnp < 0xFF )
            {
                m_dnp++;// Increment the number of null packets
                return 0;
            }
        }
    }
    // Need to send a new transport packet 
    res = add_ts_frame_base( ts );
    if( res ) m_dnp = 0;// Clear the DNP counter
    // return whether it is time to transmit a new frame
    return res;
}
DVB2::DVB2(void)
{
    // Clear the transport queue
    m_tp_q.empty();
    init_bb_randomiser();
    bch_poly_build_tables();
    build_crc8_table();
    m_dnp   = 0;// No delted null packets
    m_frame_offset_bits = 0;
    m_params_changed = 1;

	m_bitLDPC = new char[FRAME_CACHE_COUNT*FRAME_SIZE_NORMAL];
	m_bitBCH = new char[FRAME_CACHE_COUNT*FRAME_SIZE_NORMAL];
}
DVB2::~DVB2(void)
{
	delete[] m_bitLDPC;
	delete[] m_bitBCH;
}
