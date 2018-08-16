
#ifndef INCLUDED_CURE_COMMON
#define INCLUDED_CURE_COMMON

#pragma once


__device__ __forceinline__ int warp_id()
{
	return threadIdx.x / 32;
}

#endif