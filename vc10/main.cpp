
#include "DVBS2.h"
#include "DVBS2-decode.h"

#define PACKET_NUMBER	100
#define PACKET_STREAM	(PACKET_NUMBER*PACKET_SIZE)
#define CP 0x7FFF
#define PRINT_SIZE		16
#define DATA_FROM_ENC	1	// ENCODE OR FILE
#define VALUE_DIFF		60

#define DATA_FILE_NAME	"D:\\file\\data\\ldpc\\data_long\\s31.1_16apsk_34_long.dat"
void init(u8* buffer, int n);	// initialize info
void print(scmplx* c, int n, int nstart = 0);	// output encoded info
template<typename T>
void print(T* b, int n, int nstart = 0);		// output original info
int findHeader(scmplx* c, int n, int* pos);

template<typename T>
int findHeader(T* c, int n, int* pos);

void main()
{
	scmplx pl[FRAME_SIZE_NORMAL];
	short  pBuffer[FRAME_SIZE_NORMAL*2];
	printf("%d,%d,%d \n", sizeof(long), sizeof(int), sizeof(short) );
#if DATA_FROM_ENC
	DVBS2*	m_dvbs2 = new DVBS2;
	DVB2FrameFormat	dvbs2_fmt;
	//
	// DVB-S2
	//
	dvbs2_fmt.frame_type    = FRAME_NORMAL;
	dvbs2_fmt.code_rate     = CR_3_4;
	dvbs2_fmt.constellation = M_32APSK;
	dvbs2_fmt.roll_off      = RO_0_35;
	dvbs2_fmt.pilots        = 0;
	dvbs2_fmt.dummy_frame   = 0;
	dvbs2_fmt.null_deletion = 0;
	int nStatus = m_dvbs2->s2_set_configure( &dvbs2_fmt );
	if( -1 == nStatus )
	{
		printf(" mode(%d, %d) is invalid ! \n",
			dvbs2_fmt.constellation, dvbs2_fmt.code_rate );

		delete	m_dvbs2;

		return ;
	}

	u8 b[PACKET_STREAM], bRef[PACKET_STREAM];
	init( b, PACKET_STREAM );
	print( b, PACKET_STREAM );

	for (int i=0;i<PACKET_NUMBER;i++)
		m_dvbs2->s2_add_ts_frame( b + i*PACKET_SIZE );

	memcpy_s( pl, sizeof(scmplx)*FRAME_SIZE_NORMAL, 
		m_dvbs2->pl_get_frame(), sizeof(scmplx)*FRAME_SIZE_NORMAL);

	delete	m_dvbs2;
#else
	FILE *fp = fopen( DATA_FILE_NAME, "rb" );
	if( fp )
	{
		fread( pBuffer, sizeof(short), FRAME_SIZE_NORMAL*2, fp );
		memcpy_s( pl, sizeof(scmplx)*FRAME_SIZE_NORMAL, pBuffer,
			sizeof(short)*FRAME_SIZE_NORMAL*2 );
	}
	else
		printf("failed to open file %s \n",DATA_FILE_NAME );

	int position[10];
	int nCan = findHeader( pBuffer, FRAME_SIZE_NORMAL*2, position );
	printf( "\n candidate count: %d \n", nCan );
	for( int i=0; i<nCan; i++ ) {
		int pos = position[i];
		printf( "\n frame header position is: pl[%d] = (%hd, %hd, %hd) \n", 
			pos, pBuffer[pos-1], pBuffer[pos], pBuffer[pos+1] );
		print( pBuffer, PACKET_SIZE, pos );
		printf("pl[%d]: %hd, %hd \n", pos/2, pl[pos/2].re, pl[pos/2].im );
	}


#endif

	print( pBuffer, PACKET_SIZE );
	print( pl, PACKET_SIZE );

	DVBS2_DECODE*	m_dvbs2_dec = new DVBS2_DECODE;
	m_dvbs2_dec->s2_decode_ts_frame( pl );
	print( m_dvbs2_dec->getByte(), PACKET_SIZE );
	delete	m_dvbs2_dec;
}

void init(u8* buffer, int n)	// initialize info
{
	for (int i=0;i<n;i++)
		buffer[i] = i%256;
}

void print(scmplx* c, int n, int nstart/* = 0*/)	// output encoded info
{
	int nPrint = nstart+PRINT_SIZE;//n;
	for (int i=nstart;i<nPrint;i++)
		printf("%d: (%hd,%hd), (%f,%f)\n", i, c[i].re, c[i].im,
		c[i].re*1.0f/CP, c[i].im*1.0f/CP );
}

template<typename T>
void print(T* b, int n, int nstart/* = 0*/)		// output original info
{
	int nPrint = nstart+PRINT_SIZE;//n;
	for (int i=nstart;i<nPrint;i++)
		printf("%d: %d \n", i, b[i] );
}

template<typename T>
int findHeader( T* c, int n, int* pos )
{
	int i=1, j=0;
	int diff = VALUE_DIFF;
	for (; i<n-1&&j<10; i++)
		if( abs( abs(c[i]) - abs(c[i-1]) ) < diff && 
			abs( abs(c[i]) - abs(c[i+1]) ) < diff )
			pos[j++]=i;
		else;

	return j;
}
