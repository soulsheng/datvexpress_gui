
#pragma once


#include <itpp/itcomm.h>
using namespace itpp;

void convertBufferToVec( char* buffer, bvec& a );

void convertBufferToVec( double* buffer, vec& a );

void convertBufferToVec( double* buffer, cvec& a );

void convertVecToBuffer( char* buffer, bvec& a );

void convertVecToBuffer( double* buffer, vec& a );

void convertVecToBuffer( double* buffer, cvec& a );

