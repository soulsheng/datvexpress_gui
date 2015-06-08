
#include "DVBS2.h"
#include "DVBS2-decode.h"
#include "helper_timer.h"

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



	FILE *fp2 = fopen( DATA_FILE_NAME_ENC, "rb" );
	if( fp2 )
	{
		fread( pBuffer, sizeof(short), FRAME_SIZE_NORMAL*2, fp2 );
		memcpy_s( pl, sizeof(scmplx)*FRAME_SIZE_NORMAL, pBuffer,
			sizeof(short)*FRAME_SIZE_NORMAL*2 );
	}
	else
		printf("failed to open file %s \n",DATA_FILE_NAME );

	fclose( fp2 );

	print( pl, PACKET_SIZE );

	DVBS2_DECODE*	m_dvbs2_dec = new DVBS2_DECODE;
	m_dvbs2_dec->initialize();


	StopWatchInterface	*timerStep;
	sdkCreateTimer( &timerStep );
	sdkResetTimer( &timerStep );
	sdkStartTimer( &timerStep );

	m_dvbs2_dec->s2_decode_ts_frame( pl );

	sdkStopTimer( &timerStep );
	float fTime =sdkGetTimerValue( &timerStep ) ;
	int nBit = 64800;

	printf("decode time : %f \n", fTime );
	printf("decode speed : %f \n", 64800/fTime/5 );

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
