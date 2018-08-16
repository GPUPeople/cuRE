#ifndef INCLUDED_CURE_PROGRESS_QUEUE
#define INCLUDED_CURE_PROGRESS_QUEUE

#pragma once

#include <meta_utils.h>
#include <ptx_primitives.cuh>
#include <cub/cub.cuh>

#include <intrin.h>


template <unsigned int SIZE>
class ProgressQueue
{
	static_assert(static_popcnt<SIZE>::value == 1, "ProgressQueue size must be a power of two");
private:
	static constexpr int QUEUESIZE = (SIZE + 31) / 32;

	unsigned int bitfields[QUEUESIZE];
	unsigned int already_done;

public:
	__device__
	void init()
	{
		stg_cg(&already_done, 0u);
		int linid = blockIdx.x * blockDim.x + threadIdx.x;
		for (int i = linid; i < QUEUESIZE; i += blockDim.x * gridDim.x)
			stg_cg(bitfields + i, 0x0u);
	}

	__device__
	void reset()
	{
		unsigned int prev;
		//clear remaining bits
		//for (int i = 0; i < QUEUESIZE; i += blockDim.x)
		int i = 0;
		do
		{
			prev = bitfields[(already_done / 32 + threadIdx.x + i*blockDim.x) % QUEUESIZE];
			bitfields[(already_done / 32 + threadIdx.x) % QUEUESIZE] = 0;
			i += blockDim.x;
		} while (__syncthreads_or(prev));
		stg_cg(&already_done, 0u);
	}

	__device__
	void markDone(unsigned int id)
	{
		// just set the bit to one
		unsigned int withinElementOffset = id % 32;
		unsigned int element = id / 32;
		element = element % QUEUESIZE;

		atomicOr(&bitfields[element], 0x1u << withinElementOffset);
	}

	template<unsigned int NUM_THREADS>
	using CheckProgressReduce = cub::BlockReduce<unsigned int, NUM_THREADS>;

	template<unsigned int NUM_THREADS>
	using CheckProgressShared = typename CheckProgressReduce<NUM_THREADS>::TempStorage;

	template<unsigned int NUM_THREADS>
	__device__
	unsigned int checkProgressBlock(CheckProgressShared<NUM_THREADS>& sharedstorage)
	{
		__shared__ unsigned int current_already_done, new_already_done;
		current_already_done = ldg_cg(&already_done);
		__syncthreads();

		//every thread loads an uint beginning from already_done
		unsigned int myMask = ldg_cg(&bitfields[(current_already_done / 32 + threadIdx.x) % QUEUESIZE]);

		//compute first not completed position
		unsigned int firstNotCompleted = __ffs(~myMask);
		firstNotCompleted = (current_already_done & 0xFFFFFFE0u) + ((firstNotCompleted == 0) ? (NUM_THREADS * 32) : (threadIdx.x * 32 + firstNotCompleted - 1));

		//find minimum in block
		unsigned int agg_result =  CheckProgressReduce<NUM_THREADS>(sharedstorage). Reduce(firstNotCompleted, cub::Min());

		//update already_done
		if (threadIdx.x == 0)
			new_already_done = max(agg_result, atomicMax(&already_done, agg_result));
		__syncthreads();

		//unset all that are before (only entire uints)
		if (threadIdx.x < (new_already_done / 32 - current_already_done / 32))
			stg_cg(&bitfields[(current_already_done / 32 + threadIdx.x) % QUEUESIZE], 0);

		return new_already_done;
	}
};

#endif //INCLUDED_CURE_PROGRESS_QUEUE
