#ifndef DVB2_INTERLEAVE_H
#define DVB2_INTERLEAVE_H
#include <queue>
#include <deque>
#include <list>
#include <stdio.h>
using namespace std;

#include "dvb_types.h"
#include "modulatorDefinition.h"


class DVB_Interleave {

public:
	void s2_interleave( int* iframe, Bit* bframe, int& m_payload_symbols );
	void s2_b2i( int* iframe, Bit* bframe, int& m_payload_symbols );

	void initialize( DVB2FrameFormat *f );

protected:

	DVB2FrameFormat *m_format;

};

#endif
