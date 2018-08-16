


#ifndef INCLUDED_CURE_RASTERIZATION_STAGE
#define INCLUDED_CURE_RASTERIZATION_STAGE

#pragma once

#include "triangle_buffer.cuh"
#include "index_queue.cuh"
#include "progress_queue.cuh"

#include "config.h"
#include <cub/cub.cuh>
#include <ptx_primitives.cuh>
#include <bitonic_sort.cuh>
#include "instrumentation.cuh"

#ifndef RASTERIZATION_STAGE_GLOBAL
#define RASTERIZATION_STAGE_GLOBAL extern
#endif

//RASTERIZATION_STAGE_GLOBAL __device__ IndexQueue<RASTERIZER_QUEUE_SIZE, unsigned int, true, true> rasterizer_queue[NUM_BLOCKS];
//RASTERIZATION_STAGE_GLOBAL __device__ TriangleBuffer<TRIANGLE_BUFFER_SIZE, NUM_INTERPOLATORS, true> triangle_buffer;
static_assert(RASTERIZER_QUEUE_SIZE >= 2 * RASTERIZATION_CONSUME_THRESHOLD, "RASTERIZER_COSUME_THRESHOLD too restrictive... increase RASTERIZER_QUEUE_SIZE?");


template<bool PRIMITIVE_ORDER, bool USECUB = true>
struct RasterizerQueueT;

template<bool A>
struct RasterizerQueueT<false, A>
{ 
	typedef unsigned int IndexQueueType;
	MultiIndexQueue<MAX_NUM_RASTERIZERS, RASTERIZER_QUEUE_SIZE, IndexQueueType, IndexQueueAccessControl::EnumAccessControl<INDEXQUEUEATOMICS, INDEXQUEUEABORTONOVERFLOW>, -1U, TRACK_FILL_LEVEL> index_queue;

	struct SharedMemT
	{
	};

	__device__
	void newPrimitive()
	{
	}

	__device__
	void init()
	{
		index_queue.init();
	}

	__device__
	int dequeueIndexBlock(int q, unsigned int &triid, int num_threads)
	{
		return index_queue.dequeueBlock(q, &triid, num_threads);
	}

	__device__
	void completedPrimitive(unsigned int primitive_id)
	{
	}

	__device__
	void enqueue(int q, unsigned int triangle_id, unsigned int primitive_id)
	{
		index_queue.enqueue(q, triangle_id);
	}
	

	template<int NUM_THREADS>
	struct SortQueueShared
	{

	};

	template<int NUM_THREADS>
	__device__
	bool sortQueue(int q, char* shared_memory_in, volatile int * sufficientToRun)
	{
		*sufficientToRun = index_queue.size(q) >= NUM_THREADS;
		return false;
	}

	__device__
	int availableElements(int q)
	{
		return index_queue.size(q);
	}

	__device__
	int count(int q)
	{
		return index_queue.size(q);
	}
};

template<int NUM_THREADS, int SORTING_ELEMENTS, bool CUB>
struct RasterizerQueueSorter;

template<int NUM_THREADS, int SORTING_ELEMENTS>
struct RasterizerQueueSorter<NUM_THREADS, SORTING_ELEMENTS, true>
{
	typedef typename cub::BlockRadixSort<unsigned int, NUM_THREADS, SORTING_ELEMENTS, int>::TempStorage SharedT;

	__device__
	static void sort(SharedT& storage, unsigned int(&keys)[SORTING_ELEMENTS], int (&values)[SORTING_ELEMENTS], int begin_bit = 0, int end_bit = 32)
	{
		cub::BlockRadixSort<unsigned int, NUM_THREADS, SORTING_ELEMENTS, int>(storage).SortBlockedToStriped(keys, values, begin_bit, end_bit);
	}
};

template<int NUM_THREADS>
struct RasterizerQueueSorter<NUM_THREADS, 2, false>
{
	struct SharedT
	{
		unsigned int sort_keys[2 * NUM_THREADS];
		int sort_values[2 * NUM_THREADS];
	};

	__device__
	static void sort(SharedT& storage, unsigned int(&keys)[2], int(&values)[2], int begin_bit = 0, int end_bit = 32)
	{
		#pragma unroll
		for (int i = 0; i < 2; ++i)
			storage.sort_keys[threadIdx.x + i*NUM_THREADS] = keys[i],
			storage.sort_values[threadIdx.x + i*NUM_THREADS] = values[i];
		BitonicSort::sort<unsigned int, int, NUM_THREADS, true>(storage.sort_keys, storage.sort_values, threadIdx.x);
		for (int i = 0; i < 2; ++i)
			keys[i] = storage.sort_keys[threadIdx.x + i*NUM_THREADS],
			values[i] = storage.sort_values[threadIdx.x + i*NUM_THREADS];
	}
};

template<bool USECUB>
struct RasterizerQueueT<true, USECUB>
{
	static constexpr int SORTING_ELEMENTS = USECUB ? 10 : 2;
	static constexpr int TAKE_ALONG = USECUB ? 5 : 1;
	static constexpr bool REDUCE_SORTED_BITS = true;
	//static_assert(SORTING_ELEMENTS >= 2, "SortingElements must be at least two, to get a sufficient number of primitives sorted with a single run");

	typedef unsigned long long int IndexQueueType;
	typedef ProgressQueue<TRIANGLE_BUFFER_SIZE> ProgressQueueType;
	typedef MultiIndexQueue<MAX_NUM_RASTERIZERS, RASTERIZER_QUEUE_SIZE, IndexQueueType, IndexQueueAccessControl::EnumAccessControl<INDEXQUEUEATOMICS, INDEXQUEUEABORTONOVERFLOW>, -1U, TRACK_FILL_LEVEL> MultiIndexQueueType;

	MultiIndexQueueType index_queue;
	ProgressQueueType triangle_progress;

	MultiIndexQueueType::QueuePos ready[MAX_NUM_RASTERIZERS];
	//unsigned int lastReadyPrimitive[MAX_NUM_RASTERIZERS];

	template<int NUM_THREADS>
	using Sorter = RasterizerQueueSorter<NUM_THREADS, SORTING_ELEMENTS, USECUB>;

	__device__
	void newPrimitive()
	{
		triangle_progress.reset();
		//if (REDUCE_SORTED_BITS && threadIdx.x < MAX_NUM_RASTERIZERS)
		//	lastReadyPrimitive[threadIdx.x] = 0;
	}

	__device__
	void init()
	{
		index_queue.init();
		triangle_progress.init();
		if (threadIdx.x < MAX_NUM_RASTERIZERS)
			ready[threadIdx.x] = 0;
			
	}
	__device__
	int dequeueIndexBlock(int q, unsigned int &triid, int num_threads)
	{
		__shared__ int take, offset;
		unsigned long long comptriid = triid;

		if (threadIdx.x == 0)
		{
			int num = min(num_threads, availableElements(q));
			int ttake = index_queue.singleThreadReserveRead(q, num);
			offset = index_queue.singleThreadTake(q, ttake);
			take = ttake;
		}
		__syncthreads();
		if (threadIdx.x < take)
			index_queue.multiThreadRead(q, &comptriid, threadIdx.x, offset);

		triid = static_cast<unsigned int>(comptriid & 0xFFFFFFFFULL);
		return take;
	}

	__device__
	void completedPrimitive(unsigned int primitive_id)
	{
		triangle_progress.markDone(primitive_id);
	}

	__device__
	void enqueue(int q, unsigned int triangle_id, unsigned int primitive_id)
	{
		index_queue.enqueue(q, (static_cast<unsigned long long>(primitive_id) << 32) | triangle_id);
	}

	template<int NUM_THREADS>
	struct SortQueueShared
	{
		union
		{
			Sorter<NUM_THREADS>::SharedT sort_storage;
			ProgressQueueType::CheckProgressShared <NUM_THREADS> progress_storage;
		};
		cub::BlockReduce<int, NUM_THREADS>::TempStorage reduce_storage;
		MultiIndexQueueType::QueuePos rdy;
		int count, sorted, toSort, available_primitives, sortbits, lastReadyPrimitive;
	};

	template<int NUM_THREADS>
	__device__
	bool sortQueue(int q, char* shared_memory_in, volatile int * sufficientToRun)
	{
		static_assert(static_popcnt<NUM_THREADS>::value == 1, "NUM_THREADS for sorting must be a power of two");

		//Instrumentation::BlockObserver<10, 1> observer;

		SortQueueShared<NUM_THREADS>& shared_memory = *new(shared_memory_in)SortQueueShared<NUM_THREADS>;
		
		// initial check if it makes any sense to do something even
		if (threadIdx.x == 0)
		{
			shared_memory.rdy = ready[q];
			shared_memory.count = index_queue.begin(q).until(shared_memory.rdy);
			shared_memory.toSort = min(index_queue.size(q), RASTERIZER_QUEUE_SIZE) - shared_memory.count;
			*sufficientToRun = shared_memory.count >= NUM_THREADS;
		}

		__syncthreads();


		if (shared_memory.toSort == 0)
			return false;

		if (TAKE_ALONG > 1)
			if (shared_memory.count >= NUM_THREADS)
				return true;

		// update the progress queue
		unsigned int available = [&]()->unsigned int {
			Instrumentation::BlockObserver<11, 2> observer;
			return triangle_progress. template checkProgressBlock<NUM_THREADS>(shared_memory.progress_storage);
		}();

		__syncthreads();

		// get current state of the queue
		// we need to redo that, as we now only know available primitive number - if something got enqueued 
		// after we checked initally, we could miss a tringle!
		if (threadIdx.x == 0)
		{
			shared_memory.available_primitives = available;
			shared_memory.toSort = min(index_queue.size(q), RASTERIZER_QUEUE_SIZE) - shared_memory.count;
			if (REDUCE_SORTED_BITS)
			{ 
				//shared_memory.lastReadyPrimitive = lastReadyPrimitive[q];
				//shared_memory.sortbits = max(4, 32 - __clz(available - shared_memory.lastReadyPrimitive + 3));
				shared_memory.sortbits = max(4, 32 - __clz(available + 2));
			}
		}
		__syncthreads();

		// TODO: for now we ignore and do not update the sorted pointer 
		// (could be used to make sure that we dont sort if it is not needed and stop sorting ealier if nothing changes anymore)
		{
			//there is something to sort -> run from back to either sorted or ready and update ready
			Instrumentation::BlockObserver<12, 2> observer;

			int newReadyOffset = shared_memory.toSort;
			int toSort = newReadyOffset;
			
			int startOffset;

			do
			{
				startOffset = toSort - SORTING_ELEMENTS * NUM_THREADS;

				unsigned int local_keys[SORTING_ELEMENTS];
				int local_values[SORTING_ELEMENTS];

				#pragma unroll
				for (int i = 0; i < SORTING_ELEMENTS; ++i)
				{
					local_values[i] = startOffset + threadIdx.x + i*NUM_THREADS;
					if (local_values[i] >= 0)
					{
						// we can try to remap the sorting range to fewer bits
						unsigned int primitiveid = static_cast<unsigned int>(((shared_memory.rdy + local_values[i]).read(index_queue, q) >> 32) & 0xFFFFFFFFULL);
						//if (REDUCE_SORTED_BITS)
						//	local_keys[i] = 2 + min(primitiveid, shared_memory.available_primitives + 1) - shared_memory.lastReadyPrimitive;
						//else
							local_keys[i] = 1 + min(primitiveid, shared_memory.available_primitives + 1);
					}
					else
						local_keys[i] = 0;
				}

				Sorter<NUM_THREADS>::sort(shared_memory.sort_storage, local_keys, local_values, 0, REDUCE_SORTED_BITS ? shared_memory.sortbits : 32);

				// read entries
				unsigned long long int entries[SORTING_ELEMENTS];
				#pragma unroll
				for (int i = SORTING_ELEMENTS-1; i >= 0; --i)
				{
					if (local_values[i] >= 0)
					{
						entries[i] = (shared_memory.rdy + local_values[i]).read(index_queue, q);
						//if (REDUCE_SORTED_BITS)
						//{
						//	if (local_keys[i] - 2 + shared_memory.lastReadyPrimitive >= shared_memory.available_primitives)
						//		newReadyOffset = startOffset + threadIdx.x + NUM_THREADS*i;
						//}
						//else
						{ 
							if (local_keys[i] - 1 >= shared_memory.available_primitives)
								newReadyOffset = startOffset + threadIdx.x + NUM_THREADS*i;
						}
					}
				}

					__syncthreads();

				// write the entries to their new positions
				#pragma unroll
				for (int i = 0; i < SORTING_ELEMENTS; ++i)
				{
					int outoffset = startOffset + threadIdx.x + NUM_THREADS*i;
					if (outoffset >= 0)
						(shared_memory.rdy + outoffset).write(index_queue, q, entries[i]);
				}
				__syncthreads();

				toSort -= NUM_THREADS*(SORTING_ELEMENTS - TAKE_ALONG);

			} while (startOffset > 0);

			// compute new ReadOffset accross block
			newReadyOffset = cub::BlockReduce<int, NUM_THREADS>(shared_memory.reduce_storage).Reduce(min(newReadyOffset, TAKE_ALONG*NUM_THREADS), cub::Min());
			if (threadIdx.x == 0)
			{
				*sufficientToRun = (shared_memory.count + newReadyOffset) >= NUM_THREADS;
				//if (REDUCE_SORTED_BITS && newReadyOffset > 0)
				//{ 
				//	lastReadyPrimitive[q] = 1 + static_cast<unsigned int>(((shared_memory.rdy + (newReadyOffset - 1)).read(index_queue, q) >> 32) & 0xFFFFFFFFULL);
				//	ready[q] = shared_memory.rdy + newReadyOffset;
				//}
				//else
					ready[q] = shared_memory.rdy + newReadyOffset;
			}
		 }
		__syncthreads();
		return true;
	}

	__device__
	int availableElements(int q)
	{
		auto front = index_queue.begin(q);
		return front.until(ready[q]);
	}

	__device__
	int count(int q)
	{
		return index_queue.size(q);
	}
};



template <int BITS, typename = void>
struct bitmask_type_t;

template <int BITS>
struct bitmask_type_t<BITS, typename enable_if<(BITS <= 32)>::type>
{
	using type = unsigned int;
};

template <int BITS>
struct bitmask_type_t<BITS, typename enable_if<(BITS > 32 && BITS <= 64)>::type>
{
	using type = unsigned long long;
};

template <int BITS>
using bitmask_type = typename bitmask_type_t<BITS>::type;

template <int NUM_RASTERIZERS, int ACTIVE_BITS>
class VirtualRasterizers
{
	static constexpr int NUM_ELEMENTS = (NUM_RASTERIZERS + ACTIVE_BITS - 1) / ACTIVE_BITS;

	bitmask_type<ACTIVE_BITS> active[NUM_ELEMENTS];

public:
	__device__
	void init()
	{
		for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < NUM_ELEMENTS; i += gridDim.x * blockDim.x)
			active[i] = 0U;
	}

	__device__
	bool isRasterizerActive(int id) const
	{
		const unsigned int mask = 0x1U << (id % ACTIVE_BITS);
		return (ldg_cg(&active[id / ACTIVE_BITS]) & mask) != 0;
	}

	__device__
	bool setRasterizerActive(int id)
	{
		const unsigned int mask = 0x1U << (id % ACTIVE_BITS);
		bool b = (atomicOr(&active[id / ACTIVE_BITS], mask) & mask) == 0;

		//if (threadIdx.x == 0 && b)
		//	printf("%ulld %d acquired %d\n", clock64(), blockIdx.x, id);
		return b;
	}

	__device__
	void setRasterizerInactive(int id)
	{
		//if (threadIdx.x == 0)
		//	printf("%ulld %d releasing %d\n", clock64(), blockIdx.x, id);
		atomicAnd(&active[id / ACTIVE_BITS], ~(0x1U << (id % ACTIVE_BITS)));
	}
};


RASTERIZATION_STAGE_GLOBAL __device__ VirtualRasterizers<MAX_NUM_RASTERIZERS, 32> virtual_rasterizers;


typedef RasterizerQueueT<ENFORCE_PRIMITIVE_ORDER> RasterizerQueue;

RASTERIZATION_STAGE_GLOBAL __device__ RasterizerQueue rasterizer_queue;
RASTERIZATION_STAGE_GLOBAL __device__ TriangleBuffer<TRIANGLE_BUFFER_SIZE, NUM_INTERPOLATORS, TRIANGLEBUFFER_REFCOUNTING, TRACK_FILL_LEVEL> triangle_buffer;

#endif  // INCLUDED_CURE_RASTERIZATION_STAGE
