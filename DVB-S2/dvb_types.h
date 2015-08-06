#ifndef DVB_TYPES_H
#define DVB_TYPES_H

#include "inttypes.h"

typedef struct{
    short re;
    short im;
}scmplx;


//#define scmplx fftw_complex

typedef struct{
    double re;
    double im;
}dcmplx;

typedef struct{
    float re;
    float im;
}fcmplx;

typedef struct{
	int ts_gs;
	int sis_mis;
	int ccm_acm;
	int issyi;
	int npd;
	int ro;
	int isi;
	int upl;
	int dfl;
	int sync;
	int syncd;
}BBHeader;

typedef int Bit;

// The number of useable and stuff bits in a frame
typedef struct{
	int data_bits;
	int stuff_bits;
}FrameBits;

typedef struct{
	int frame_type;
	int code_rate;
	int roll_off;
	int constellation;
	int pilots;
	int dummy_frame;
	int null_deletion;
	int interface;
	int broadcasting;
	// Calculated information, not used by caller
	BBHeader bb_header;
	int kldpc;
	int kbch;
	int nldpc;
	int q_val;
	int bch_code;
	int useable_data_bits;
	int padding_bits;
	int total_bits;
	int nr_tps;
	int nr_tps_bits;
}DVB2FrameFormat;


#endif // DVB_TYPES_H
