


#ifndef INCLUDED_FREEPIPE_DEPTHSTORAGE
#define INCLUDED_FREEPIPE_DEPTHSTORAGE

#pragma once

extern "C"
{
	__global__ void resetDepthStorage(float* buffer, float value, int width, int height)
	{
		int x = blockDim.x * blockIdx.x + threadIdx.x;
		int y = blockDim.y * blockIdx.y + threadIdx.y;
		if (x < width && y < height)
			buffer[x + y * width] = value;
	}
}
#endif
