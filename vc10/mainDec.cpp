
#include "DVBS2.h"
#include "DVBS2-decode.h"
#include "helper_timer.h"

#define PACKET_NUMBER	100
#define PACKET_STREAM	(PACKET_NUMBER*PACKET_SIZE)
#define CP 0x7FFF
#define PRINT_SIZE		188 * 20

#define DATA_FILE_NAME_ENC	"../data/s31.1_32apsk_34_long.dat"

void init(u8* buffer, int n);	// initialize info
void print(scmplx* c, int n, int nstart = 0);	// output encoded info
template<typename T>
void print(T* b, /*int n, */int nstart = 0, int nsize=PRINT_SIZE);		// output original info
template<typename T>
int verify(T* b, /*int n, */int nstart = 0);		// verify original info


void main()
{
	scmplx* pl = new scmplx[FRAME_SIZE_NORMAL*FRAME_CACHE_COUNT];
	short  pBuffer[FRAME_SIZE_NORMAL*2];


	int nFrameCount = 0;
	FILE *fp2 = fopen( DATA_FILE_NAME_ENC, "rb" );
	if( !fp2 )
		printf("failed to open file %s \n",DATA_FILE_NAME_ENC );
	
	fread( &nFrameCount, sizeof(int), 1, fp2 );

	for ( int i = 0; i<nFrameCount; i++ )
	{
		fread( pBuffer, sizeof(short), FRAME_SIZE_NORMAL*2, fp2 );
		memcpy_s( pl + i*FRAME_SIZE_NORMAL, sizeof(scmplx)*FRAME_SIZE_NORMAL, pBuffer,
			sizeof(short)*FRAME_SIZE_NORMAL*2 );
	}

	fclose( fp2 );

	printf("\nframe count : %d ... ... \n\n", nFrameCount );	

	DVBS2_DECODE*	m_dvbs2_dec = new DVBS2_DECODE;
	m_dvbs2_dec->initialize();


	StopWatchInterface	*timerStep;
	sdkCreateTimer( &timerStep );

	
	sdkResetTimer( &timerStep );
	sdkStartTimer( &timerStep );

#if 0	// test step time
	nFrameCount = 1;
	m_dvbs2_dec->decode_ts_frame( pl );
#else
	m_dvbs2_dec->decode_ts_frame( pl, nFrameCount );
#endif

	sdkStopTimer( &timerStep );
	float fTime =sdkGetTimerValue( &timerStep )/nFrameCount ;
	
	int nSymbol = m_dvbs2_dec->s2_get_n_symbol();

	printf("\ndecode time : %.2f ms\n", fTime );	// 27 ms, 529(d) -> 4 ms
	printf("decode speed : %f MBd/s \n\n", nSymbol/fTime * 0.001f );

	for ( int i = 0;i<nFrameCount;i++ )
	{
		//print( m_dvbs2_dec->getByte(i), 0, 200 );
		int nError = verify( m_dvbs2_dec->getByte(i) );
		if( 0 == nError )
			printf("\nframe %d succeed to decode\n\n",i);
		else
			printf("\nframe %d failed to decode %d bits\n\n",i, nError);
		printf("____________________________________________\n");
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
int verify(T* b, /*int n, */int nstart/* = 0*/)		// output original info
{
	int nPrint = nstart+PRINT_SIZE;//n;
	int nPositionFirstError = -1;
	bool bFirstError = true;
	int nErrorCount = 0;
	for (int i=nstart;i<nPrint;i++)
	{
		if( /*i%PACKET_SIZE &&*/ (i+b[1]-1)%256 != b[i] )
		{
			if( bFirstError )
			{
				nPositionFirstError = i;
				bFirstError = false;
				continue;
			}
			
			if( (i - nPositionFirstError)%PACKET_SIZE == 0 )
				continue;

			printf("\nerror bit [%d]=%d \n", i, b[i] );
			printf(", the bit should be %d \n", (i+b[1]-1)%256 );
			nErrorCount ++;
		}
	}
	return nErrorCount;
}
