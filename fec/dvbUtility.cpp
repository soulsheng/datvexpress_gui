
#include "dvbUtility.h"


void convertBufferToVec( char* buffer, bvec& a )
{
	for (int i = 0; i< a.size(); i++ )
		a._elem(i) = buffer[i] ;
}


void convertBufferToVec( double* buffer, vec& a )
{
	for (int i = 0; i< a.size(); i++ )
		a._elem(i) = buffer[i] ;
}

void convertBufferToVec( double* buffer, cvec& a )
{
	for (int i = 0; i< a.size(); i++ ){
		a._elem(i).real( buffer[i*2] );
		a._elem(i).imag( buffer[i*2+1] );
	}
}

void convertVecToBuffer( char* buffer, bvec& a )
{
	for (int i = 0; i< a.size(); i++ )
		buffer[i] = a._elem(i).value() ;
}

void convertVecToBuffer( double* buffer, vec& a )
{
	for (int i = 0; i< a.size(); i++ )
		buffer[i] = a._elem(i) ;
}

void convertVecToBuffer( double* buffer, cvec& a )
{
	for (int i = 0; i< a.size(); i++ ){
		buffer[i*2]		= a._elem(i).real( );
		buffer[i*2+1]	= a._elem(i).imag( );
	}
}

//! Maximum value of vector
int max(int *v, int N)
{
	int tmp = v[0];
	for (int i = 1; i < N; i++)
		if (v[i] > tmp)
			tmp = v[i];
	return tmp;
}

//! Minimum value of vector
int min(int *v, int N)
{
	int tmp = v[0];
	for (int i = 1; i < N; i++)
		if (v[i] < tmp)
			tmp = v[i];
	return tmp;
}


void	writeFile(int nvar, int ncheck, int nmaxX1, int nmaxX2, char* filename)
{
	FILE* fp;
	fp = fopen( filename, "wb" );
	if( !fp )
		return;

	fwrite( &nvar, sizeof(int), 1, fp );
	fwrite( &ncheck, sizeof(int), 1, fp );
	fwrite( &nmaxX1, sizeof(int), 1, fp );
	fwrite( &nmaxX2, sizeof(int), 1, fp );

	fclose( fp );
}

void	readFile(int& nvar, int& ncheck, int& nmaxX1, int& nmaxX2, char* filename)
{
	FILE* fp;
	fp = fopen( filename, "rb" );
	if( !fp )
		return;

	fread( &nvar, sizeof(int), 1, fp );
	fread( &ncheck, sizeof(int), 1, fp );
	fread( &nmaxX1, sizeof(int), 1, fp );
	fread( &nmaxX2, sizeof(int), 1, fp );

	fclose( fp );
}
