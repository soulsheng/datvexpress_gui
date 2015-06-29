
#include "DVBS2.h"
#include "DVBS2-decode.h"

#define PACKET_NUMBER	100
#define PACKET_STREAM	(PACKET_NUMBER*PACKET_SIZE)
#define CP 0x7FFF
#define PRINT_SIZE		16
#define DATA_FROM_ENC	1	// ENCODE OR FILE
#define VALUE_DIFF		60

#define DATA_FILE_NAME	"D:\\file\\data\\ldpc\\data_long\\s31.1_16apsk_34_long.dat"
#define DATA_FILE_NAME_ENC	"../data/s31.1_32apsk_34_long.dat"

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
	{
		if( m_dvbs2->s2_add_ts_frame( b + i*PACKET_SIZE ) )
			/*break*/;
	}

	FILE *fp = fopen( DATA_FILE_NAME_ENC, "wb" );
	if( fp )
	{
		int nFrameCount = m_dvbs2->get_frame_count();

		fwrite( &nFrameCount, sizeof(int), 1, fp );

		for ( int i = 0; i<nFrameCount; i++ )
		{

			memcpy_s( pl, sizeof(scmplx)*FRAME_SIZE_NORMAL, 
				m_dvbs2->pl_get_frame(i), sizeof(scmplx)*FRAME_SIZE_NORMAL);

			memcpy_s(  pBuffer,	sizeof(short)*FRAME_SIZE_NORMAL*2, 
				pl, sizeof(scmplx)*FRAME_SIZE_NORMAL );

			fwrite( pBuffer, sizeof(short), FRAME_SIZE_NORMAL*2, fp );
		}

		fclose( fp );
	}
	else
		printf("failed to open file %s \n",DATA_FILE_NAME );

	delete	m_dvbs2;

	print( pl, PACKET_SIZE );

	printf("pl_symbol(90): \n");
	print( pl, PACKET_SIZE, 90 );

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
