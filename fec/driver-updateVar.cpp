
#include "driver-updateVar.cuh"
#include "helper_timer.h"
#include <cuda_runtime.h>
#include <iostream>
using	namespace	std;


void main()
{
	bool bStatus = false;

	StopWatchInterface*	timer;
	sdkCreateTimer( &timer );

	driverUpdataVar ldpc;

	sdkStartTimer( &timer );

	if( !ldpc.launch() )
		cout << "Failed to launch" << endl;

	cudaDeviceSynchronize();
	sdkStopTimer( &timer );
	cout << "time of kernel updateVar is : " << sdkGetTimerValue( &timer ) << endl;

	if( !ldpc.verify() )
		cout << "Failed to verify" << endl;
	else
		cout << "Succeed to launch cuda kernel and verify that result is right" << endl;

	cudaDeviceReset();
	//system( "pause" );

}

