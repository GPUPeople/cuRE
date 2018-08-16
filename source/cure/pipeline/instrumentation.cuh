


#ifndef INCLUDED_CURE_INSTRUMENTATION
#define INCLUDED_CURE_INSTRUMENTATION

#pragma once

#include <ptx_primitives.cuh>
#include <instrumentation.cuh>

#include "config.h"

#ifndef INSTRUMENTATION_GLOBAL
#define INSTRUMENTATION_GLOBAL extern
#endif


extern "C"
{
	INSTRUMENTATION_GLOBAL __device__ unsigned char mp_index[MAX_NUM_BLOCKS];
}

INSTRUMENTATION_GLOBAL __device__ Instrumentation::Instrumentation<MAX_NUM_BLOCKS, Timers::NUM_TIMERS, Timers::template Enabled> instrumentation;

namespace Instrumentation
{
	template <int id, int LEVEL>
	struct BlockObserver
	{
		__device__ BlockObserver() { instrumentation.enter<id, LEVEL>(); }
		__device__ ~BlockObserver() { instrumentation.leave<id, LEVEL>(); }
	};

	template <int LEVEL>
	struct BlockObserver<0, LEVEL>
	{
		__device__ BlockObserver() { mp_index[blockIdx.x] = smid(); instrumentation.enter<0, LEVEL>(); }
		__device__ ~BlockObserver() { instrumentation.leave<0, LEVEL>(); }
	};
}

#endif  // INCLUDED_CURE_INSTRUMENTATION
