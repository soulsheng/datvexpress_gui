
#include "dvbUtility.h"


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

void	readFile(int& nCodeword, int& nAlpha, int& nGrid, char* filename)
{
	FILE* fp;
	fp = fopen( filename, "rb" );
	if( !fp )
		return;

	fread( &nCodeword, sizeof(int), 1, fp );
	fread( &nAlpha, sizeof(int), 1, fp );
	fread( &nGrid, sizeof(int), 1, fp );

	fclose( fp );
}

void	writeFile(int& nCodeword, int& nAlpha, int& nGrid, char* filename)
{
	FILE* fp;
	fp = fopen( filename, "wb" );
	if( !fp )
		return;

	fwrite( &nCodeword, sizeof(int), 1, fp );
	fwrite( &nAlpha, sizeof(int), 1, fp );
	fwrite( &nGrid, sizeof(int), 1, fp );

	fclose( fp );
}

void writeFile( std::vector<int>& paramSize, char* filename )
{
	FILE* fp;
	fp = fopen( filename, "wb" );
	if( !fp )
		return;

	for (int i=0;i<paramSize.size();i++)
		fwrite( &paramSize[i], sizeof(int), 1, fp );

	fclose( fp );
}

void writeFile( std::vector<int*>& paramSize, char* filename )
{
	FILE* fp;
	fp = fopen( filename, "wb" );
	if( !fp )
		return;

	for (int i=0;i<paramSize.size();i++)
		fwrite( paramSize[i], sizeof(int), 1, fp );

	fclose( fp );
}

void readFile( std::vector<int>& paramSize, char* filename )
{
	FILE* fp;
	fp = fopen( filename, "rb" );
	if( !fp )
		return;

	while( feof(fp) )
	{	
		int val;
		fwrite( &val, sizeof(int), 1, fp );
		paramSize.push_back(val);
	}

	fclose( fp );
}

void readFile( std::vector<int*>& paramSize, char* filename )
{
	FILE* fp;
	fp = fopen( filename, "rb" );
	if( !fp )
		return;

	for (int i=0;i<paramSize.size();i++)
		fread( paramSize[i], sizeof(int), 1, fp );

	fclose( fp );
}

int lfsr( unsigned long int *seed )
{
	int b,c;

	b = ( ((*seed) & (1 << 31) ) >> 31 ) ;

	c =   ((*seed) & 1) ^ ( ((*seed) & (1 << 1)) >> 1 ) ^ ( ((*seed) & (1 << 21)) >> 21 ) ^ b ;

	(*seed) = ((*seed) << 1) | c;

	return(b);
}