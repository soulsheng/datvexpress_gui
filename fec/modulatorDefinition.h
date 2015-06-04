
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

// Code rates
#define CR_1_4 0
#define CR_1_3 1
#define CR_2_5 2
#define CR_1_2 3
#define CR_3_5 4
#define CR_2_3 5
#define CR_3_4 6
#define CR_4_5 7
#define CR_5_6 8
#define CR_8_9 9
#define CR_9_10 10

#define FRAME_NORMAL 0x00
#define FRAME_SHORT  0x10


// Constellation
#define M_QPSK   0
#define M_8PSK   1
#define M_16APSK 2
#define M_32APSK 3
#define M_CONST_NUMBER	 4

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

#define		CODE_RATE_DEFAULT	CR_1_2
#define		FRAME_TYPE_DEFAULT	FRAME_SHORT