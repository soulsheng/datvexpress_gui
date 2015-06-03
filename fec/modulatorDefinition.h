
#pragma once

#include <vector>
#include <map>
#include <iostream>
using namespace std;


#define BB_HEADER_LENGTH_BITS 72

// BB HEADER fields
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

#define FRAME_SIZE_NORMAL 64800
#define FRAME_SIZE_SHORT  16200

// BCH Code
#define BCH_CODE_N8  0
#define BCH_CODE_N10 1
#define BCH_CODE_N12 2
#define BCH_CODE_S12 3

#define LDPC_ENCODE_TABLE_LENGTH (FRAME_SIZE_NORMAL * 10)

enum	MOD_TYPE
{
	MOD_DEFAULT,//	0
	MOD_BPSK,	//	1
	MOD_QPSK,	//	2
	MOD_8PSK,	//	3
	MOD_16APSK,	//	4
	MOD_32APSK,	//	5
	MOD_COUNT	//	6
};

enum CODE_RATE
{
	C1_4,
	C1_3,
	C2_5,
	C1_2,
	C3_5,
	C2_3,
	C3_4,
	C4_5,
	C5_6,
	C8_9,
	C9_10
};

enum FRAME_TYPE
{
	FECFRAME_NORMAL,
	FECFRAME_SHORT
};


enum Rolloff_Factor
{
	RO_0_35 = 0,
	RO_0_25,
	RO_0_20,
	RO_RESERVED,
	RO_0_15,
	RO_0_10,
	RO_0_05,
};

#define CP 0x7FFF

#define		CODE_RATE_DEFAULT	C1_2
#define		FRAME_TYPE_DEFAULT	FECFRAME_SHORT