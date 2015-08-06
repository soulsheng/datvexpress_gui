#ifndef DVB_TYPES_H
#define DVB_TYPES_H

#include "inttypes.h"
#include "modulatorDefinition.h"

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

	int configure()
	{
		int error = 0;
		DVB2FrameFormat*f = this;
		if( f->broadcasting )
		{
			// Set standard parametrs for broadcasting
			//f->frame_type        = FRAME_NORMAL;
			f->bb_header.ts_gs   = TS_GS_TRANSPORT;
			f->bb_header.sis_mis = SIS_MIS_SINGLE;
			f->bb_header.ccm_acm = CCM;
			f->bb_header.issyi   = 0;
			f->bb_header.npd     = 0;
			f->bb_header.upl     = 188*8;
			f->bb_header.sync    = 0x47;
		}
		f->bb_header.ro = f->roll_off;
		// Fill in the mode specific values and bit lengths
		if( f->frame_type == FRAME_NORMAL )
		{
			f->nldpc = 64800;
			f->bch_code = BCH_CODE_N12;
			// Apply code rate
			switch(f->code_rate )
			{
			case CR_1_4:
				f->q_val = 135;
				f->kbch  = 16008;
				f->bch_code = BCH_CODE_N12;
				break;
			case CR_1_3:
				f->q_val = 120;
				f->kbch  = 21408;
				f->bch_code = BCH_CODE_N12;
				break;
			case CR_2_5:
				f->q_val = 108;
				f->kbch  = 25728;
				f->bch_code = BCH_CODE_N12;
				break;
			case CR_1_2:
				f->q_val = 90;
				f->kbch  = 32208;
				f->bch_code = BCH_CODE_N12;
				break;
			case CR_3_5:
				f->q_val = 72;
				f->kbch  = 38688;
				f->bch_code = BCH_CODE_N12;
				break;
			case CR_2_3:
				f->q_val = 60;
				f->kbch  = 43040;
				f->bch_code = BCH_CODE_N10;
				break;
			case CR_3_4:
				f->q_val = 45;
				f->kbch  = 48408;
				f->bch_code = BCH_CODE_N12;
				break;
			case CR_4_5:
				f->q_val = 36;
				f->kbch  = 51648;
				f->bch_code = BCH_CODE_N12;
				break;
			case CR_5_6:
				f->q_val = 30;
				f->kbch  = 53840;
				f->bch_code = BCH_CODE_N10;
				break;
			case CR_8_9:
				f->q_val = 20;
				f->kbch  = 57472;
				f->bch_code = BCH_CODE_N8;
				break;
			case CR_9_10:
				f->q_val = 18;
				f->kbch  = 58192;
				f->bch_code = BCH_CODE_N8;
				break;
			default:
				printf("Configuration error DVB2\n");
				error = -1;
				break;
			}
		}

		if( f->frame_type == FRAME_SHORT )
		{
			f->nldpc = 16200;
			f->bch_code = BCH_CODE_S12;
			// Apply code rate
			switch(f->code_rate )
			{
			case CR_1_4:
				f->q_val = 36;
				f->kbch  = 3072;
				break;
			case CR_1_3:
				f->q_val = 30;
				f->kbch  = 5232;
				break;
			case CR_2_5:
				f->q_val = 27;
				f->kbch  = 6312;
				break;
			case CR_1_2:
				f->q_val = 25;
				f->kbch  = 7032;
				break;
			case CR_3_5:
				f->q_val = 18;
				f->kbch  = 9552;
				break;
			case CR_2_3:
				f->q_val = 15;
				f->kbch  = 10632;
				break;
			case CR_3_4:
				f->q_val = 12;
				f->kbch  = 11712;
				break;
			case CR_4_5:
				f->q_val = 10;
				f->kbch  = 12432;
				break;
			case CR_5_6:
				f->q_val = 8;
				f->kbch  = 13152;
				break;
			case CR_8_9:
				f->q_val = 5;
				f->kbch  = 14232;
				break;
			case CR_9_10:
				error = 1;
				f->kbch  = 0;
				break;
			default:
				printf("Configuration error DVB2\n");
				error = -1;
				break;
			}
		}

		int bch_bits = 0;
		if( f->frame_type == FRAME_NORMAL )
		{
			switch(f->code_rate )
			{
			case CR_2_3:
			case CR_5_6:
				bch_bits = 160;
				break;
			case CR_8_9:
			case CR_9_10:
				bch_bits = 128;
				break;
			default:
				bch_bits = 192;
			}
		}
		else
		{
			bch_bits = 168;
		}

		if( error == 0 )
		{
			// Length of the user packets
			f->bb_header.upl  = 188*8;
			// Payload length
			f->bb_header.dfl = f->kbch - 80;
			// Transport packet sync
			f->bb_header.sync = 0x47;
			// Start of LDPC bits
			f->kldpc = f->kbch + bch_bits;
			// Number of padding bits required (not used)
			f->padding_bits = 0;
			// Number of useable data bits (not used)
			f->useable_data_bits = f->kbch - 80;
		}

		return error;
	}
}DVB2FrameFormat;


#endif // DVB_TYPES_H
