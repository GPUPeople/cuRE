
#ifndef INCLUDED_BITONIC_SORT
#define INCLUDED_BITONIC_SORT

#pragma once

#include "ptx_primitives.cuh"

namespace BitonicSort
{
	template<class Key, class Value>
	__device__ inline void bitonic_comp(Key& key_a, Key& key_b,
		Value& val_a, Value& val_b,
		bool dir)
	{
		if ((key_a != key_b) && (key_a > key_b) == dir)
		{
			//swap
			Key kT = key_a;
			key_a = key_b;
			key_b = kT;

			Value vT = val_a;
			val_a = val_b;
			val_b = vT;
		}
	}


	template<class Key, class Value, int THREADS, bool Dir>
	__device__ void sort(Key* keys, Value* values, uint linId)
	{
		for (uint size = 2; size < 2 * THREADS; size <<= 1)
		{
			//bitonic merge
			bool d = Dir ^ ((linId & (size / 2)) != 0);
			for (uint stride = size / 2; stride > 0; stride >>= 1)
			{
				__syncthreads();
				uint pos = 2 * linId - (linId & (stride - 1));
				bitonic_comp(keys[pos], keys[pos + stride],
					values[pos], values[pos + stride],
					d);
			}
		}

		//final merge
		for (uint stride = THREADS; stride > 0; stride >>= 1)
		{
			__syncthreads();
			uint pos = 2 * linId - (linId & (stride - 1));
			bitonic_comp(keys[pos], keys[pos + stride],
				values[pos], values[pos + stride],
				Dir);
		}
		__syncthreads();
	}
}



#endif  // INCLUDED_BITONIC_SORT
