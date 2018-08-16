

#ifndef CURE_TOOLS_BITONICSOURT_INCLUDED
#define CURE_TOOLS_BITONICSOURT_INCLUDED

namespace FreePipe
{
	namespace Sort
	{
		template <class Key, class Value>
		__device__ inline void bitonic_comp(volatile Key& key_a, volatile Key& key_b, volatile Value& val_a, volatile Value& val_b, bool dir)
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

		template <class Key, class Value, int Elements, int Threads, bool Dir>
		__device__ void bitonic(volatile Key* keys, volatile Value* values, int tid)
		{
			for (int size = 2; size < Elements; size <<= 1)
			{
				//bitonic merge
				
				for (int stride = size / 2; stride > 0; stride >>= 1)
				{
					__syncthreads();
					#pragma unroll
					for (int linOffset = 0; linOffset < Elements / 2; linOffset += Threads)
					{
						int linId = linOffset + tid;
						if (linId < Elements / 2)
						{
							bool d = Dir ^ ((linId & (size / 2)) != 0);
							int pos = 2 * linId - (linId & (stride - 1));
							bitonic_comp(keys[pos], keys[pos + stride], values[pos], values[pos + stride], d);
						}
					}
				}
			}

			//final merge
			for (int stride = Elements / 2; stride > 0; stride >>= 1)
			{
				__syncthreads();
				#pragma unroll
				for (int linOffset = 0; linOffset < Elements / 2; linOffset += Threads)
				{
					int linId = linOffset + tid;
					if (linId < Elements / 2)
					{
						int pos = 2 * linId - (linId & (stride - 1));
						bitonic_comp(keys[pos], keys[pos + stride], values[pos], values[pos + stride], Dir);
					}
				}
			}
			__syncthreads();
		}
		
	}
}

#endif // CURE_TOOLS_BITONICSOURT_INCLUDED
