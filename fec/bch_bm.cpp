#include <stdlib.h>
#include <conio.h>
#include <iostream>
#include <time.h>
#include <dos.h>
#include <stdio.h>
#include <string.h>
#include <fstream>
#include <math.h>

#include "bch_bm.h"

#include "helper_timer.h"
#define		TIME_STEP		6	

int BCH_BM::lfsr(unsigned long int *seed)
{
	int b,c;

	b = ( ((*seed) & (1 << 31) ) >> 31 ) ;

	c =   ((*seed) & 1) ^ ( ((*seed) & (1 << 1)) >> 1 ) ^ ( ((*seed) & (1 << 21)) >> 21 ) ^ b ;

	(*seed) = ((*seed) << 1) | c;

	return(b);
}

/****************************************************************************/
/*********************** Message generator **********************************/
/***************************************************************************/

void BCH_BM::message_gen(int n,int k, unsigned long int  *seed, char* message)
{
	int i;
    // Message bits pseudo random generation
	for (i=n-1;i>=n-k;i--)
		message[i] = lfsr(seed);
	// Zero padding
	for(i = 0; i < n-k; i++)
		message[i] = 0;
}

/****************************************************************************/
/*********************** Polynomial Generators *****************************/
/***************************************************************************/
// Note: only used by algorithm emulating the serial architecture (n-clock cycles)

const unsigned int gen12[] =
{1,1,1,0,0,1,1,1,1,0,1,0,1,0,1,0,0,1,0,0,0,0,0,0,0,1,1,0,0,1,1,0,
 1,1,1,0,1,1,1,1,1,0,1,0,0,0,0,1,1,1,1,0,0,0,1,0,1,1,0,0,0,0,0,0,
 1,0,0,1,0,0,0,1,0,0,0,1,0,0,0,0,1,0,1,0,1,1,0,0,0,0,1,1,1,0,1,1,
 0,0,0,1,1,0,1,1,0,0,1,1,0,1,0,0,1,1,1,1,0,0,1,1,0,0,0,0,1,0,1,0,
 0,0,1,1,1,0,0,0,1,0,0,0,1,0,1,0,0,0,1,1,1,0,1,0,0,0,1,0,0,0,0,1,
 1,1,0,0,0,0,0,1,0,1,1,1,0,0,0,0,0,1,1,0,0,1,0,0,0,1,1,1,0,0,1,0,
 1};
// i.e. gen(x) = a_0*x^0 + a_1*x^1 + ... + a_(r-1)*x^(r-1) + a_r*x^r

const unsigned int gen10[] =
{1,0,0,0,1,0,0,1,1,0,1,0,0,1,1,0,1,1,0,1,1,1,0,1,0,0,0,1,1,1,0,1,
 1,0,0,0,0,0,0,0,1,1,0,0,0,1,0,0,1,0,0,0,1,0,1,1,1,1,1,1,0,1,1,1,
 1,1,0,0,0,0,0,0,1,1,1,0,1,0,1,0,0,0,0,1,1,1,1,0,0,1,0,1,0,1,1,0,
 1,1,1,1,1,0,0,0,1,1,0,0,1,1,0,0,0,1,0,1,0,1,0,0,0,0,1,1,1,1,1,1,
 1,0,1,1,0,1,1,1,0,0,1,1,0,0,0,0,1,0,1,0,1,0,0,0,0,0,0,0,0,1,1,0,
 1};

const unsigned int gen8[] =
{1,1,0,1,0,1,0,0,0,1,1,0,0,1,1,0,1,0,0,1,1,1,1,1,0,0,1,0,0,0,0,0,
 1,0,1,0,1,1,1,0,1,0,1,1,0,1,1,0,0,0,1,1,1,1,1,1,1,0,0,1,1,0,0,0,
 1,0,1,1,1,1,0,1,1,1,1,0,1,0,0,1,1,1,1,0,0,1,0,0,1,0,0,0,1,1,1,0,
 1,1,1,1,1,0,1,0,1,0,1,0,0,1,0,0,1,1,1,0,0,0,0,0,0,0,1,1,1,0,0,0,
 1};


const unsigned int gen12_s[] = 
{1,0,1,0,0,1,0,1,1,0,1,0,0,0,0,0,1,0,0,1,1,0,0,0,1,0,0,0,1,0,1,1,
1,1,1,0,1,0,1,1,1,1,1,0,0,1,1,1,1,1,1,1,0,0,0,1,0,1,0,0,1,0,1,0,
1,0,0,1,0,1,1,0,0,0,0,0,1,0,0,1,1,1,0,0,0,1,0,1,1,1,0,0,0,1,0,0,
1,0,1,1,0,0,1,1,0,1,0,0,0,1,1,0,0,1,0,0,1,1,0,1,1,0,0,1,0,1,1,0,
0,0,0,1,1,0,0,1,0,1,0,1,0,1,1,1,1,1,0,1,1,0,1,1,0,1,0,0,0,1,1,0,
0,0,0,0,0,0,1,0,
1};	// short frame

/****************************************************************************/
/*********************** Serial BCH encoder ********************************/
/***************************************************************************/

void BCH_BM::encode( char* message, char* codeword)
{

	const unsigned int *g;
	
	int mem,app,i,j;

/***************  Mode Selection (t-error-correction) ***********************/

	switch(n-k) {
	case 192:
		g = gen12;
		break;
	case 160:
		g = gen10;
		break;
	case 128:
		g = gen8;
		break;
	case 168:
		g = gen12_s;
		break;
	default:
		fprintf(stdout,"Error:simulation aborted!\n");
		fprintf(stdout,"Please insert a n-k couple provided by DVB-S2 FEC\n");
		exit(0);
	}
	
	memset( reg, 0, sizeof(int)*MAXR );

/*********************** Encoding serial algorithm ********************************/
/**************************   n clock ticks **************************************/

/************************* Computing remainder **********************************/

	for (i=n-1; i>=0; i--)
	{
		mem=reg [n-k-1];
		for (j=n-k-2; j>=0; j--)
		{
			app=mem & g[j+1];
			reg[j+1]=reg[j]^app;
		}

		reg[0]= message[i]^(mem & g[0]);

	}

/*********************** Codeword in systematic form ********************************/

	for (i=n-1;i>=n-k;i--)
		codeword[i] = message[i];
	for (i=n-k-1; i >=0; i--)
		codeword[i] = reg[i];

}


/****************************************************************************/
/*********************** Creation of GF(2^m)  *******************************/
/*********************** useful tables        *******************************/
/***************************************************************************/

void BCH_BM::gfField(int m, // Base 2 logarithm of cardinality of the Field
			 int poly, // primitive polynomial of the Field in decimal form
			 int* powAlpha,
			 int* indexAlpha)
{
	int reg,	// this integer of 32 bits, masked by a sequence of m ones,
				// contains the elements of Galois Field
		tmp,i;
	// sequence of m ones
	int mask = (1<<m) -1;  // 1(m) Bit Masking

	powAlpha[0] = 1;
	indexAlpha[0] = - 1; // we set -1
	indexAlpha[1] = 0;

	for (i = 0, reg = 1; i < (1<<m)-2; i++)
	{
			tmp = (int)(reg & (1<<(m-1))); // Get the MSB
            reg <<= 1;   // Register shifted
            if( tmp) { //
				reg ^= poly;
				//
				reg &= mask;
			}
			// Step-by-step writing of the tables
			powAlpha[i+1] = (int) reg;
			indexAlpha[(int)reg] = i+1;
    }


}


/****************************************************************************/
/*********************** Error detection   *******************************/
/***************************************************************************/

bool BCH_BM::error_detection(  char* codeword)
{
	int tCapacity = 0;
	if ( code_type == FRAME_NORMAL )
		tCapacity = t(n,k) + DRIFT;
	else
		tCapacity = 12 + DRIFT;

#if 1
	bool syn = false;
	for(int i = 0; i < tCapacity*2; i++)
	{
		S[i] = 0;
		for(int j = 0; j < n; j++)
		{
			if(codeword[j])
				S[i] ^= powAlpha[((i+1)*j)%MAXN];
		}

		S[i] = indexAlpha[S[i]];

		if(S[i] != -1)
			syn = true;

	}
#else// use block

	int block = BLOCK_DIM;
	int grid = (n+BLOCK_DIM-1)/BLOCK_DIM ;

	bool syn = false;
	for(int i = 0; i < tCapacity*2; i++)
	{
		S[i] = 0;

		for ( int bID = 0; bID < grid; bID++ )
		{
			int		s_powAlpha[BLOCK_DIM] ;
			char	s_codeword[BLOCK_DIM] ;
			memset( s_powAlpha, 0, sizeof(int) * BLOCK_DIM );
			memset( s_codeword, 0, sizeof(char) * BLOCK_DIM );

			for( int tID = 0; tID < block; tID++ )
			{
				int j = bID * block + tID ;

				if( j>=n )
					break;

				s_codeword[ tID ] = codeword[ j ];

				if(s_codeword[ tID ])
					s_powAlpha[ tID ] = powAlpha[ ((i+1)*j)%MAXN ];
				else
					s_powAlpha[ tID ] = 0;


			}

			for( int offset = block / 2; offset>=1; offset /= 2 )
			{
				for( int tID = 0; tID < block; tID++ )
				{
					if( tID < offset )
						s_powAlpha[ tID ] ^= s_powAlpha[ tID + offset ];
				}
			}

			S[i] ^= s_powAlpha[0];
		}


		S[i] = indexAlpha[S[i]];

		if(S[i] != -1)
			syn = true;

	}

#endif

	return syn;

}


/****************************************************************************/
/*********************** Error correction   *******************************/
/***************************************************************************/

void BCH_BM::BerlMass( )

{
	int t2 = 2*tCapacity;
	int j,L,l,i;
	int d, dm, tmp;
	int *T, *c, *p, *lambda;
	// Allocation and initialization
	// Auto-Regressive-Filter coefficients computed at the previous step
	p = (int*) calloc(t2,sizeof(int));
	// Auto-Regressive-Filter coefficients computed at the current step
	c = (int*) calloc(t2,sizeof(int));
	// Temporary array
	T = (int*) calloc(t2,sizeof(int));
	// error location array (found by Chien Search)
	//el = (int*) calloc(t2,sizeof(int));
	// Error polynomial locator
	lambda = (int*) calloc(t2,sizeof(int));

	memset( el, -1, sizeof(int)*MAXT );

	// Inizialization step
	c[0] = 1;
	p[0] = 1;
	L = 0;
	l = 1;
	dm = 1;

/*********** Berlekamp-Massey Algorithm *******************/
	for (j = 0; j < t2; j++)
	{
		// Discrepancy computation
		if(S[j] == -1)
			d = 0;
		else
			d = powAlpha[S[j]];
		for(i = 1; i <= L;i++)
			if(S[j-i] >= 0 && c[i] > 0)
			d ^= powAlpha[(indexAlpha[c[i]]+ S[j-i])%MAXN];
			// exponential rule

		if( d == 0)
		{
			l++;
		}
		else
		{
			if(2*L > j)
			{
				for( i = l; i <t2; i++)
				{
					if(p[i-l] != 0)
						c[i] ^= powAlpha[(indexAlpha[d]-indexAlpha[dm]+indexAlpha[p[i-l]]+MAXN)%MAXN];
				}
				l++;
			}
			else
			{
				for( i = 0; i < t2; i++)
					T[i] = c[i];
				for( i = l; i <t2; i++)
				{
					if(p[i-l] != 0)
						c[i] ^= powAlpha[(indexAlpha[d]-indexAlpha[dm]+indexAlpha[p[i-l]]+MAXN)%MAXN];
				}
				L = j-L+1;
				for( i = 0; i < t2; i++)
					p[i] = T[i];
				dm = d;
				l = 1;
			}

		}
	}



/********** Storing of error locator polynomial coefficient **********/
	for(i = 0; i <=L; i++)
	{
		// Error storing
		lambda[i] = indexAlpha[c[i]];

	}

/**************    Chien search   **************************/
/*******************   Roots searching  ***********************/

	int kk = 0;
	for(i = 0; i < MAXN; i++)
	{
		for(j = 1, tmp = 0; j <=L; j++)
			tmp ^= powAlpha[(lambda[j]+i*j)%MAXN];
		if (tmp == 1)
			// roots inversion give the error locations
			el[kk++] = (MAXN-i)%MAXN;

	}// 2.4 ms 
	


	free(T); free(c); free(p); free(lambda); //free(el);

}


/*********************** print msg and code  *******************************/
void BCH_BM::printNK( char* message, char* codeword, int length )
{
	std::cout << std::endl << "msg:" << std::endl;
	int nMax = n-k+length-1;
	for (int i=nMax;i>=n-k;i--)
		std::cout << (int)message[i] << " ";

	std::cout << std::endl << "code:" << std::endl;
	for (int i=nMax;i>=n-k;i--)
		std::cout << (int)codeword[i] << " ";

	std::cout << std::endl;
}

void BCH_BM::BCH_final_dec(  char* message, char* codeword )
{
	for (int i=0;i<k;i++)
		message[i] = codeword[i];
}

bool BCH_BM::verifyResult(  char* message, char* messageRef )
{
	bool bSuccess = true;
	for (int i=n-1;i>=n-k;i--)	{
		if( message[i] != messageRef[i])	{
			bSuccess = false;
			break;
		}
	}

	return bSuccess;
}

BCH_BM::BCH_BM()
	:mNormal(16), mShort(14)
{
	// Allocation and initialization of the tables of the Galois Field
	powAlphaNormal = (int *)calloc((1<<mNormal), sizeof(int));
	indexAlphaNormal = (int *)calloc((1<<mNormal), sizeof(int));

	// Galois Field Creation
	gfField(mNormal, 32+8+4+1, powAlphaNormal, indexAlphaNormal);

	powAlphaShort = (int *)calloc((1<<mShort), sizeof(int));
	indexAlphaShort = (int *)calloc((1<<mShort), sizeof(int));

	// Galois Field Creation
	gfField(mShort, 32+8+2+1, powAlphaShort, indexAlphaShort);

}

BCH_BM::~BCH_BM()
{
	release();
}

void BCH_BM::initialize()
{
	el = (int*) calloc(MAXT*2,sizeof(int));
	reg = (int*)calloc(MAXR,sizeof(int));
}

void BCH_BM::release()
{
	//free(powAlpha);	free(indexAlpha);
	free( el );
	free( reg );
}

void BCH_BM::decode(  char* messageRecv, char* codeword )
{
	float	timerStepValue[TIME_STEP];

	int nTimeStep = 0;
	StopWatchInterface	*timerStep;

	sdkCreateTimer( &timerStep );
	sdkStartTimer( &timerStep );

	bool errCode = false;

#ifdef	USE_GPU
	errCode = m_bch_gpu.error_detection(codeword);// 1ms
#else
	errCode = error_detection(codeword);// 7 ms
#endif

	sdkStopTimer( &timerStep );
	timerStepValue[nTimeStep++] = sdkGetTimerValue( &timerStep );

	sdkResetTimer( &timerStep );
	sdkStartTimer( &timerStep );

	if( errCode ) {
		fprintf(stdout,"Errors detected!\nDecoding by Berlekamp-Massey algorithm.....\n");

		BerlMass();

		sdkStopTimer( &timerStep );
		timerStepValue[nTimeStep++] = sdkGetTimerValue( &timerStep );// 3 ms

		sdkResetTimer( &timerStep );
		sdkStartTimer( &timerStep );

		bool success = true;
		fprintf(stdout,"\nPosition of errors detected:\n");
		for(int i = 0; i <MAXT; i++) 
		{
			if ( -1 != el[i] )
			{
				codeword[ el[i] ] ^= 1;
				fprintf(stdout,"%d\t",el[i]);
			}
		}

		if(success) {
		fprintf(stdout,"\nSuccessful decoding!\n----------------------\n");};
		
		sdkStopTimer( &timerStep );
		timerStepValue[nTimeStep++] = sdkGetTimerValue( &timerStep );// 0.8 ms

		sdkResetTimer( &timerStep );
		sdkStartTimer( &timerStep );

	}
	else
		fprintf(stdout,"\n\nNo errors detected!\n------------------------------\n");


	BCH_final_dec(messageRecv, codeword);

	sdkStopTimer( &timerStep );
	timerStepValue[nTimeStep++] = sdkGetTimerValue( &timerStep );// 0.06 ms

#if 0// cost time 0.7ms/cout 
	for (int i=0;i<TIME_STEP;i++)
	{
		cout  << "timerStepValue[ " << i << " ] = "<< timerStepValue[i] << " ms, " << endl;
	}
#endif
}

void BCH_BM::setCode( int rate, int type )
{
	code_rate = rate;
	code_type = type;

	if( FRAME_NORMAL == code_type ) {
		switch( code_rate )
		{
		case CR_1_4:	
			n = 16200; k=16008;
			break;
		case CR_1_3:	
			n = 21600; k=21408; 
			break;
		case CR_2_5:	
			n = 25920; k=25728; 
			break;
		case CR_1_2:	
			n = 32400; k=32208; 
			break;
		case CR_3_5:	
			n = 38880; k=38688; 
			break;
		case CR_2_3:	
			n = 43200; k=43040; 
			break;
		case CR_3_4:	
			n = 48600; k=48408; 
			break;
		case CR_4_5:	
			n = 51840; k=51648; 
			break;
		case CR_5_6:	
			n = 54000; k=53840; 
			break;
		case CR_8_9:	
			n = 57600; k=57472; 
			break;
		case CR_9_10:	
			n = 58320; k=58192;
			break;
		default:
			break;
		}// switch

		m = 16;
		powAlpha = powAlphaNormal;
		indexAlpha = indexAlphaNormal;

	}// if normal
	else	// short
	{
		switch( code_rate )
		{
		case CR_1_4:	
			n = 3240; k=3072;
			break;
		case CR_1_3:	
			n = 5400; k=5232; 
			break;
		case CR_2_5:	
			n = 6480; k=6312; 
			break;
		case CR_1_2:	
			n = 7200; k=7032; 
			break;
		case CR_3_5:	
			n = 9720; k=9552; 
			break;
		case CR_2_3:	
			n = 10800; k=10632; 
			break;
		case CR_3_4:	
			n = 11880; k=11712; 
			break;
		case CR_4_5:	
			n = 12600; k=12432; 
			break;
		case CR_5_6:	
			n = 13320; k=13152; 
			break;
		case CR_8_9:	
			n = 14400; k=14232; 
			break;
		default:
			break;
		}// switch

		m = 14;
		powAlpha = powAlphaShort;
		indexAlpha = indexAlphaShort;

	}// else short

	MAXN = (1<<m)-1;


	if ( code_type == FRAME_NORMAL )
		tCapacity = t(n,k) + DRIFT;
	else
		tCapacity = 12 + DRIFT;

	int nS = (MAXT + DRIFT)*2;

#ifdef USE_GPU
	m_bch_gpu.initialize( powAlpha, indexAlpha, mNormal, 
		S, nS, n, tCapacity, MAXN );
#endif
}

int BCH_BM::getN( )
{
	return n;
}

int BCH_BM::getK( )
{
	return k;
}
