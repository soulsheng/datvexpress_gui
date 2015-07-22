
#include "driver-softBit.cuh"
#include "helper_timer.h"
#include <cuda_runtime.h>
#include <iostream>
using	namespace	std;


void main()
{
	bool bStatus = false;

	StopWatchInterface*	timer;
	sdkCreateTimer( &timer );

	driverSoftBit _kernel;

	sdkStartTimer( &timer );

	if( !_kernel.launch() )
		cout << "Failed to launch" << endl;

	cudaDeviceSynchronize();
	sdkStopTimer( &timer );
	cout << "time of kernel softBit is : " << sdkGetTimerValue( &timer ) << endl;

	if( !_kernel.verify() )
		cout << "Failed to verify" << endl;
	else
		cout << "Succeed to launch cuda kernel and verify that result is right" << endl;

	cudaDeviceReset();
	//system( "pause" );

}

