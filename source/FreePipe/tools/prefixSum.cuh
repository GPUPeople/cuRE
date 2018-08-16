

#ifndef CURE_TOOLS_PREFIXSUM_INCLUDED
#define CURE_TOOLS_PREFIXSUM_INCLUDED

namespace FreePipe
{
	namespace Prefix
	{

		template <class T, int Elements, int Threads>
		__device__ T exclusive(volatile T* data, int tid)
		{
			int offset = 1;
			#pragma unroll
			for (int d = Elements / 2; d > 0; d /= 2)
			{
				__syncthreads();
				#pragma unroll
				for (int loff = 0; loff < d; loff += Threads)
				{ 
					int lid = loff + tid;
					if (lid < d)
					{
						int ai = offset*(2 * lid + 1) - 1;
						int bi = offset*(2 * lid + 2) - 1;
						data[bi] += data[ai];
					}
				}
				offset *= 2;
			}

			__syncthreads();

			T num = data[Elements - 1];
			if (tid == 0)
				data[Elements - 1] = 0;

			#pragma unroll
			for (int d = 1; d < Elements; d *= 2)
			{
				offset /= 2;
				__syncthreads();

				#pragma unroll
				for (int loff = 0; loff < d; loff += Threads)
				{
					int lid = loff + tid;
					if (lid < d)
					{
						int ai = offset*(2 * lid + 1) - 1;
						int bi = offset*(2 * lid + 2) - 1;
						float t = data[ai];
						data[ai] = data[bi];
						data[bi] += t;
					}
				}
			}
			__syncthreads();
			return num;
		}
		
	}
}

#endif // CURE_TOOLS_PREFIXSUM_INCLUDED
