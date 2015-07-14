
#include "driver-chienSearch.cuh"
#include "helper_timer.h"
#include <cuda_runtime.h>
#include <iostream>
using	namespace	std;


void main()
{
	bool bStatus = false;

	StopWatchInterface*	timer;
	sdkCreateTimer( &timer );

	driverChienSearch driverKernel;

	sdkStartTimer( &timer );

	if( !driverKernel.launch() )
		cout << "Failed to launch" << endl;

	cudaDeviceSynchronize();
	sdkStopTimer( &timer );
	cout << "time of kernel chienSearch is : " << sdkGetTimerValue( &timer ) << endl;

	if( !driverKernel.verify() )
		cout << "Failed to verify" << endl;
	else
		cout << "Succeed to launch cuda kernel and verify that result is right" << endl;

	cudaDeviceReset();
	//system( "pause" );

}

