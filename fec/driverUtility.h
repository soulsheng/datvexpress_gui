
#pragma once

#include <iostream>
using namespace std;

#define		WRITE_FILE_FOR_DRIVER	1

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
