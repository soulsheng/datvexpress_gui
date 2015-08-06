#ifndef DVB2_H
#define DVB2_H
#include <queue>
#include <deque>
#include <list>
#include <stdio.h>
#include "dvb_types.h"
#include "modulatorDefinition.h"
#include "bch_bm.h"
#include "dvb2_ldpc_encode.h"

using namespace std;

typedef unsigned int u32;
typedef unsigned char u8;

// BB HEADER fileds
#define TS_GS_TRANSPORT 3
#define TS_GS_GENERIC_PACKETIZED 0
#define TS_GS_GENERIC_CONTINUOUS 1
#define TS_GS_RESERVED 2

#define SIS_MIS_SINGLE 1
#define SIS_MIS_MULTIPLE 0

#define CCM 1
#define ACM 0

#define ISSYI_ACTIVE 1
#define ISSYI_NOT_ACTIVE 0

#define NPD_ACTIVE 1
#define NPD_NOT_ACTIVE 0
#if 0
// Rolloff
#define RO_0_35 0
#define RO_0_25 1
#define RO_0_20 2
#define RO_RESERVED 3
#endif

//#define FRAME_SIZE_NORMAL 64800
//#define FRAME_SIZE_SHORT  16200
//#define LDPC_ENCODE_TABLE_LENGTH (FRAME_SIZE_NORMAL*10)



#define BB_HEADER_LENGTH_BITS 72
#define CRC8_LENGTH_BITS 8

#define PADDING_LENGTH 200

#define		FRAME_CACHE_COUNT	20

//Interface
#define M_ACM 0
#define M_CCM 1
#define M_VCM 2




class DVB2{

public:

protected:
    Bit *m_frame, *m_frameMulti;
    DVB2FrameFormat m_format[2];
    Bit m_padding[PADDING_LENGTH];
        int m_frame_offset_bits;
        int m_params_changed;

protected:
    int m_bbheader[BB_HEADER_LENGTH_BITS+CRC8_LENGTH_BITS];
    int m_bb_randomise[FRAME_SIZE_NORMAL];
    u32 m_poly_n_8[4];
    u32 m_poly_n_10[5];
    u32 m_poly_n_12[6];
    u32 m_poly_s_12[6];
    u8  m_crc_tab[256];
    u8  m_dnp; // Deleted null packet
    // Transport packet queue
    queue <u8> m_tp_q;

	Ldpc_encode	m_ldpc;


	char	*m_bitLDPC;
	char	*m_bitBCH;
	BCH_BM	bch;

    void bb_randomise(void);
    void init_scrambler(void);
    void init_bb_randomiser(void);

    int add_transport_packet( u8 *pkt, Bit *b );
    void build_crc8_table( void );
    u8	 calc_crc8( u8 *b, int len );
    int add_crc8_bits( Bit *in, int length );
    void unpack_transport_packet_add_crc( u8 *ts );
    void add_bbheader( void );
    void poly_reverse( int *pin, int *pout, int len );
    void poly_pack( const int *pin, u32* pout, int len );
    int poly_mult( const int *ina, int lena, const int *inb, int lenb, int *out );
    void bch_poly_build_tables( void );
    void reg_4_shift( u32 *sr );
    void reg_5_shift( u32 *sr );
    void reg_6_shift( u32 *sr );
    Bit bch_n_8_encode(  Bit *in, int len );
    Bit bch_n_10_encode( Bit *in, int len );
    Bit bch_n_12_encode( Bit *in, int len );
    Bit bch_s_12_encode( Bit *in, int len );
    int bch_encode( void );
    int add_ts_frame_base( u8 *ts );

    void base_end_of_frame_actions( DVB2FrameFormat *f );
protected:
    int  set_configure( DVB2FrameFormat *f );
    void get_configure( DVB2FrameFormat *f );
    int  next_ts_frame_base( u8 *ts );
public:
    // encode a new transport packet
//    virtual void physical(void);
    DVB2();
    ~DVB2();
};

#endif
