#ifndef DVB2_LDPC_H
#define DVB2_LDPC_H
#include <queue>
#include <deque>
#include <list>
#include <stdio.h>
using namespace std;

#include "dvb_types.h"
#include "modulatorDefinition.h"

typedef int Bit;

typedef struct{
    int table_length;
    Bit d[LDPC_ENCODE_TABLE_LENGTH];
    Bit p[LDPC_ENCODE_TABLE_LENGTH];
}Ldpc_encode_table;


#define BB_HEADER_LENGTH_BITS 72
#define CRC8_LENGTH_BITS 8

#define PADDING_LENGTH 200

#define		FRAME_CACHE_COUNT	20

//Interface
#define M_ACM 0
#define M_CCM 1
#define M_VCM 2


class Ldpc_encode {

public:
	void ldpc_lookup_generate( DVB2FrameFormat *pFormat );
	void ldpc_encode( Bit *m_frame );


protected:

    // LDPC tables
    const static int ldpc_tab_1_4N[45][13];
    const static int ldpc_tab_1_3N[60][13];
    const static int ldpc_tab_2_5N[72][13];
    const static int ldpc_tab_1_2N[90][9];
    const static int ldpc_tab_3_5N[108][13];
    const static int ldpc_tab_2_3N[120][14];
    const static int ldpc_tab_3_4N[135][13];
    const static int ldpc_tab_4_5N[144][12];
    const static int ldpc_tab_5_6N[150][14];
    const static int ldpc_tab_8_9N[160][5];
    const static int ldpc_tab_9_10N[162][5];
    const static int ldpc_tab_1_4S[9][13];
    const static int ldpc_tab_1_3S[15][13];
    const static int ldpc_tab_2_5S[18][13];
    const static int ldpc_tab_1_2S[20][9];
    const static int ldpc_tab_3_5S[27][13];
    const static int ldpc_tab_2_3S[30][14];
    const static int ldpc_tab_3_4S[33][13];
    const static int ldpc_tab_4_5S[35][4];
    const static int ldpc_tab_5_6S[37][14];
    const static int ldpc_tab_8_9S[40][5];

	Ldpc_encode_table m_ldpc_encode;

	DVB2FrameFormat *m_format;

};

#endif
