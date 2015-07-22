
#pragma once


#define		FILENAME_IT		"../data/random_3_6_16200.it"
#define		FILENAME_IT34		"../data/dvbs2_r34.it"
#define		FILENAME_ALIST	"../data/dvbs2_r34.alist"
#define		FILENAME_CODE_DOWNLOAD	"../data/s311_16apsk_34_long.dat"
#define		FILENAME_IT12	"../data/dvbs2_r12.it"
#define		FILENAME_IT34	"../data/dvbs2_r34.it"

#define		COUNT_REPEAT_DEF	10	// repeat time 
#define		SIZE_PACKET		188

#define		N_BCH			31
#define		T_BCH			2
#define		K_BCH			21


#define		REMOVE_NOISE		1

#if REMOVE_NOISE
#define		EBNO			20
#else
#define		EBNO			12.6//10 2-2.2	3-5.6	4-8.9	5-12.4
#endif

#define		REMOVE_BCH			0
#define		SHORT_BCH			0
#define		MOD_TYPE_DEFAULT	4

#define		WRITE_FILE_FOR_DRIVER	0

#include <iostream>
#include <vector>
using namespace std;

//! Maximum value of vector
int max(int *v, int N);
//! Minimum value of vector
int min(int *v, int N);

void	writeFile(int nvar, int ncheck, int nmaxX1, int nmaxX2, char* filename);
void	readFile(int& nvar, int& ncheck, int& nmaxX1, int& nmaxX2, char* filename);
void	writeFile(int& nCodeword, int& nAlpha, int& nGrid, char* filename);
void	readFile(int& nCodeword, int& nAlpha, int& nGrid, char* filename);

void	writeFile(std::vector<int>& paramSize, char* filename);
void	readFile( std::vector<int>& paramSize, char* filename );
void	readFile( std::vector<int*>& paramSize, char* filename );

template <typename T>
void 	readArray(T* pArray, int nSize, char* strFileName)
{
	FILE* fp = NULL;
	fp = fopen( strFileName, "rb" );
	if(fp == NULL)
	{
		printf("failed to open: %s!\n", strFileName);
	}
	fread( pArray, sizeof(T), nSize, fp);
	fclose(fp);
}

template <typename T>
void	writeArray(T* pArray, int nSize, char* strFileName)
{
	FILE* fp = NULL;
	fp = fopen( strFileName, "wb" );
	if(fp == NULL)
	{
		printf("failed to open: %s!\n", strFileName);
	}
	fwrite( pArray, sizeof(T), nSize, fp);
	fclose(fp);
}
