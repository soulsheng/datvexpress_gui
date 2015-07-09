
#include "DVBS2.h"
#include "DVBS2-decode.h"
#include "helper_timer.h"

#define PACKET_NUMBER	100
#define PACKET_STREAM	(PACKET_NUMBER*PACKET_SIZE)
#define CP 0x7FFF
#define PRINT_SIZE		188 * 20
#define DATA_FROM_ENC	1	// ENCODE OR FILE
#define VALUE_DIFF		60

#define DATA_FILE_NAME	"D:\\file\\data\\ldpc\\data_long\\s31.1_16apsk_34_long.dat"
#define DATA_FILE_NAME_ENC	"../data/s31.1_32apsk_34_long.dat"

void init(u8* buffer, int n);	// initialize info
void print(scmplx* c, int n, int nstart = 0);	// output encoded info
template<typename T>
void print(T* b, /*int n, */int nstart = 0, int nsize=PRINT_SIZE);		// output original info
template<typename T>
bool verify(T* b, /*int n, */int nstart = 0);		// verify original info
int findHeader(scmplx* c, int n, int* pos);

template<typename T>
int findHeader(T* c, int n, int* pos);

void main()
{
	scmplx* pl = new scmplx[FRAME_SIZE_NORMAL*FRAME_CACHE_COUNT];
	short  pBuffer[FRAME_SIZE_NORMAL*2];
	printf("%d,%d,%d \n", sizeof(long), sizeof(int), sizeof(short) );


	int nFrameCount = 0;
	FILE *fp2 = fopen( DATA_FILE_NAME_ENC, "rb" );
	if( !fp2 )
		printf("failed to open file %s \n",DATA_FILE_NAME );
	
	fread( &nFrameCount, sizeof(int), 1, fp2 );

	for ( int i = 0; i<nFrameCount; i++ )
	{
		fread( pBuffer, sizeof(short), FRAME_SIZE_NORMAL*2, fp2 );
		memcpy_s( pl + i*FRAME_SIZE_NORMAL, sizeof(scmplx)*FRAME_SIZE_NORMAL, pBuffer,
			sizeof(short)*FRAME_SIZE_NORMAL*2 );
	}

	fclose( fp2 );

	print( pl, PACKET_SIZE );

	DVBS2_DECODE*	m_dvbs2_dec = new DVBS2_DECODE;
	m_dvbs2_dec->initialize();


	StopWatchInterface	*timerStep;
	sdkCreateTimer( &timerStep );

	{
	
	sdkResetTimer( &timerStep );
	sdkStartTimer( &timerStep );
	 
	m_dvbs2_dec->decode_ts_frame( pl, nFrameCount );

	sdkStopTimer( &timerStep );
	float fTime =sdkGetTimerValue( &timerStep )/nFrameCount ;
	
	int nSymbol = m_dvbs2_dec->s2_get_n_symbol();

	printf("decode time : %f \n", fTime );	// 27 ms, 529(d)
	printf("decode speed : %f MBd/s \n", nSymbol/fTime * 0.001f );

	for ( int i = 0;i<nFrameCount;i++ )
	{
	//print( m_dvbs2_dec->getByte() );
	if( verify( m_dvbs2_dec->getByte(i) ) )
		printf("succeed \n");
	else
		printf("failed \n");
	}
	}

	delete	m_dvbs2_dec;
	free( pl );
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
void print(T* b, /*int n, */int nstart/* = 0*/, int nsize)		// output original info
{
	int nPrint = nstart+nsize;//n;
	for (int i=nstart;i<nPrint;i++)
		printf("%d: %d \n", i, b[i] );
}


template<typename T>
bool verify(T* b, /*int n, */int nstart/* = 0*/)		// output original info
{
	bool bResult = true;
	int nPrint = nstart+PRINT_SIZE;//n;
	for (int i=nstart;i<nPrint;i++)
	{
		if( /*i%PACKET_SIZE &&*/ (i+b[1]-1)%256 != b[i] )
		{
			printf("\n%d: %d \n", i, b[i] );
			bResult = false;
		}
	}
	return bResult;
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
