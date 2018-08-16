


#ifndef INCLUDED_WORK_ASSIGNMENT
#define INCLUDED_WORK_ASSIGNMENT

#pragma once

#include <cub/cub.cuh>

template<int NUM_WARPS>
class BlockWorkAssignmentBase
{
protected:
	static constexpr int NUM_THREADS = NUM_WARPS * WARP_SIZE;
public:
	typedef cub::BlockScan<int, NUM_THREADS> SimpleScanT;

protected:
	__device__
	static void computeOffsets(int* work_sum, SimpleScanT::TempStorage& sum_space, int thread_work_count)
	{
		int output, aggregate;
		SimpleScanT(sum_space).ExclusiveSum(thread_work_count, output, aggregate);
		work_sum[threadIdx.x + 1] = output;
		work_sum[0] = 0;
		work_sum[NUM_THREADS + 1] = aggregate;
	}

	__device__
	static int assignWorkAllThreads(int* work_sum, int* work_offsets, SimpleScanT::TempStorage& sum_space, int& aggregate)
	{
		// clear work offsets
		work_offsets[threadIdx.x] = 0;
		__syncthreads();

		// compute which thread should start with a given work element
		int set_offset = work_sum[threadIdx.x + 1];

		// set the entry for the thread to the work id
		if (work_sum[threadIdx.x+2] != set_offset && set_offset < NUM_THREADS)
			work_offsets[set_offset] = threadIdx.x;

		__syncthreads();
		// read my offset (can be the right offset or zero as only the first one will have the right per triangle)
		int element_offset = work_offsets[threadIdx.x];

		SimpleScanT(sum_space). template InclusiveScan<cub::Max>(element_offset, element_offset, cub::Max(), aggregate);
		return element_offset;
	}

	__device__
	static int computeLocalWorkOffset(int* work_sum, int element_offset, int tid, int& inelement_tid)
	{
		// every thread gets its element offset and computes its offset within the element 
		//int within_element_forward = tid - work_sum[element_offset + 1];

		// run from back to front so we can just decrese the count iif there are not enough warps for a triangle
		int within_element_backward = work_sum[element_offset + 2] - tid - 1;
		inelement_tid = tid - work_sum[element_offset + 1];
		return within_element_backward;
	}

	__device__
	static bool pullWorkInternal(int* work_sum, int* work_offsets, SimpleScanT::TempStorage& sum_space, int& element_offset, int& within_element_offset, int& numWork, int& firstOffset, int& in_element_tid)
	{
		int aggregate;
		element_offset = assignWorkAllThreads(work_sum, work_offsets, sum_space, aggregate);
		within_element_offset = computeLocalWorkOffset(work_sum, element_offset, threadIdx.x, in_element_tid);

		numWork = min(NUM_THREADS, work_sum[NUM_THREADS + 1]);
		firstOffset = work_offsets[0];

		__syncthreads();

		// update counts
		work_sum[threadIdx.x + 2] = max(0, work_sum[threadIdx.x + 2] - NUM_THREADS);
		__syncthreads();

		// note that we might have more threads than active elements, then the within_element_offset is negative..
		return within_element_offset >= 0;
	}
};


template<int NUM_WARPS, bool PARTIAL_TAKEOUT>
class BlockWorkAssignment;

template<int NUM_WARPS>
class BlockWorkAssignment<NUM_WARPS, false> : public BlockWorkAssignmentBase<NUM_WARPS>
{
public:
	struct SharedMemT
	{
		int work_sum[NUM_THREADS + 2];
	};
	struct SharedTempMemT
	{
		SimpleScanT::TempStorage tempstorage;
		int work_offsets[ NUM_THREADS];
	};

	__device__
	static void prepare(SharedMemT& shared_memory, SharedTempMemT& shared_temp_memory, int thread_work_count)
	{
		computeOffsets(shared_memory.work_sum, shared_temp_memory.tempstorage, thread_work_count);
	}

	__device__
	static int availableWork(SharedMemT& shared_memory)
	{
		return const_cast<volatile int*>(shared_memory.work_sum)[NUM_THREADS + 1];
	}

	__device__
	static bool pullWorkThreads(SharedMemT& shared_memory, SharedTempMemT& shared_temp_memory, int& element_offset, int& within_element_offset)
	{
		int unused, unused2, unused3;
		return pullWorkInternal(shared_memory.work_sum, shared_temp_memory.work_offsets, shared_temp_memory.tempstorage, element_offset, within_element_offset, unused, unused2, unused3);
	}

	__device__
	static bool pullWorkThreads(SharedMemT& shared_memory, SharedTempMemT& shared_temp_memory, int& element_offset, int& within_element_offset, int& sumwork)
	{
		int unused, unused2;
		return pullWorkInternal(shared_memory.work_sum, shared_temp_memory.work_offsets, shared_temp_memory.tempstorage, element_offset, within_element_offset, sumwork, unused, unused2);
	}

	__device__
	static bool pullWorkThreads(SharedMemT& shared_memory, SharedTempMemT& shared_temp_memory, int& element_offset, int& within_element_offset, int& sumwork, int& firstOffset)
	{
		int unused;
		return pullWorkInternal(shared_memory.work_sum, shared_temp_memory.work_offsets, shared_temp_memory.tempstorage, element_offset, within_element_offset, sumwork, firstOffset, unused);
	}

	__device__
	static bool pullWorkThreads(SharedMemT& shared_memory, SharedTempMemT& shared_temp_memory, int& element_offset, int& within_element_offset, int& sumwork, int& firstOffset, int& inElementOffset)
	{
		return pullWorkInternal(shared_memory.work_sum, shared_temp_memory.work_offsets, shared_temp_memory.tempstorage, element_offset, within_element_offset, sumwork, firstOffset, inElementOffset);
	}
	
};


template<int NUM_WARPS>
class BlockWorkAssignment<NUM_WARPS, true> : public BlockWorkAssignmentBase<NUM_WARPS>
{
public:
	struct SharedMemT
	{
		int work_sum[NUM_THREADS + 2];
		int work_offsets[NUM_THREADS];
		int lastTaken;
	};
	struct SharedTempMemT
	{
		SimpleScanT::TempStorage tempstorage;
	};

	__device__
	static void prepare(SharedMemT& shared_memory, SharedTempMemT& shared_temp_memory, int thread_work_count)
	{
		computeOffsets(shared_memory.work_sum, shared_temp_memory.tempstorage, thread_work_count);
	}

	__device__
	static int availableWork(SharedMemT& shared_memory)
	{
		return const_cast<volatile int*>(shared_memory.work_sum)[NUM_THREADS + 1];
	}

	__device__
	static bool pullWorkThreads(SharedMemT& shared_memory, SharedTempMemT& shared_temp_memory, int& element_offset, int& within_element_offset)
	{
		int unused, unused2, unused3;
		return pullWorkInternal(shared_memory.work_sum, shared_memory.work_offsets, shared_temp_memory.tempstorage, element_offset, within_element_offset, unused, unused2, unused3);
	}

	__device__
	static bool pullWorkThreads(SharedMemT& shared_memory, SharedTempMemT& shared_temp_memory, int& element_offset, int& within_element_offset, int& sumwork)
	{
		int unused, unused2;
		return pullWorkInternal(shared_memory.work_sum, shared_memory.work_offsets, shared_temp_memory.tempstorage, element_offset, within_element_offset, sumwork, unused, unused2);
	}
	

	__device__
	static bool pullWorkThreads(SharedMemT& shared_memory, SharedTempMemT& shared_temp_memory, int& element_offset, int& within_element_offset, int& sumwork, int& firstOffset)
	{
		int unused;
		return pullWorkInternal(shared_memory.work_sum, shared_memory.work_offsets, shared_temp_memory.tempstorage, element_offset, within_element_offset, sumwork, firstOffset, unused);
	}

	__device__
	static bool pullWorkThreads(SharedMemT& shared_memory, SharedTempMemT& shared_temp_memory, int& element_offset, int& within_element_offset, int& sumwork, int& firstOffset, int& inElementOffset)
	{
		return pullWorkInternal(shared_memory.work_sum, shared_memory.work_offsets, shared_temp_memory.tempstorage, element_offset, within_element_offset, sumwork, firstOffset, inElementOffset);
	}

	__device__
	static bool prepareConsistentWorkThreads(SharedMemT& shared_memory, SharedTempMemT& shared_temp_memory)
	{
		int aggregate;
		int element_offset = assignWorkAllThreads(shared_memory.work_sum, shared_memory.work_offsets, shared_temp_memory.tempstorage, aggregate);
		__syncthreads();
		shared_memory.work_offsets[threadIdx.x] = element_offset;
		shared_memory.lastTaken = 0;
		return aggregate > 0;
	}


	__device__
	static bool takeOutConsistentWorkThreads(SharedMemT& shared_memory, int& element_offset, int& within_element_offset)
	{
		int unused;
		return takeOutConsistentWorkThreads(shared_memory, element_offset, within_element_offset, unused);
	}

	__device__
	static bool takeOutConsistentWorkThreads(int id, int takeOut, SharedMemT& shared_memory, int& element_offset, int& within_element_offset, int& numwork)
	{
		__syncthreads();
		int takeOutId = shared_memory.lastTaken + id;
		element_offset = shared_memory.work_offsets[takeOutId];
		int unused;
		within_element_offset = computeLocalWorkOffset(shared_memory.work_sum, element_offset, takeOutId, unused);
		numwork = min(NUM_THREADS,shared_memory.work_sum[NUM_THREADS + 1]) - shared_memory.lastTaken;
		__syncthreads();
		if (threadIdx.x == 0)
			shared_memory.lastTaken += takeOut;

		return within_element_offset >= 0;
	}

	__device__
	static int hasTakenWork(SharedMemT& shared_memory)
	{
		return shared_memory.lastTaken;
	}

	__device__
	static void removeTakenWorkThreads(int taken, SharedMemT& shared_memory)
	{
		// update counts
		int new_work_sum = max(0,shared_memory.work_sum[threadIdx.x + 2] - taken);
		shared_memory.work_sum[threadIdx.x + 2] = new_work_sum;
		__syncthreads();
	}
};



template<int NUM_THREADS>
class BlockWorkAssignmentOld
{
	static constexpr int NUM_WARPS = NUM_THREADS / 32;
public:
	//static constexpr size_t SHARED_MEMORY = 2 * NUM_THREADS * sizeof(int);
	//static constexpr size_t SHARED_TEMP_MEMORY = 1 * NUM_THREADS * sizeof(int)+sizeof(SimpleScanT::TempStorage);
	typedef cub::BlockScan<int, NUM_THREADS> SimpleScanT;

	struct SharedMemT
	{
		int work_count[NUM_THREADS];
		int work_sum[NUM_THREADS];
	};

	struct SharedTempMemT
	{
		SimpleScanT::TempStorage tempstorage;
		struct
		{
			int work_offsets[NUM_THREADS];
		};
	};
private:

	__device__
	static void computeOffsets(SharedMemT& shared_memory, SharedTempMemT& shared_temp_memory, int thread_work_count)
	{
		int res;
		SimpleScanT(shared_temp_memory.tempstorage).InclusiveSum(thread_work_count, res);
		shared_memory.work_sum[threadIdx.x] = res;
	}

	__device__
	static int2 assignWorkAllThreads(int* work_count, int* work_sum, int* work_offsets, SimpleScanT::TempStorage& sum_space)
	{
		// clear work offsets
		work_offsets[threadIdx.x] = 0;
		__syncthreads();

		// compute which thread should start with a given work element
		int set_offset = work_sum[threadIdx.x] - work_count[threadIdx.x];

		// set the entry for the thread to the work id
		if (work_count[threadIdx.x] > 0 && set_offset < NUM_THREADS)
			work_offsets[set_offset] = threadIdx.x;

		__syncthreads();
		// read my offset (can be the right offset or zero as only the first one will have the right per triangle)
		int element_offset = work_offsets[threadIdx.x];

		int agg;
		SimpleScanT(sum_space). template InclusiveScan<cub::Max>(element_offset, element_offset, cub::Max(), agg);


		// every thread gets its triangle offset and computes its offset within the triangle 
		int my_element_start = work_sum[element_offset] - work_count[element_offset];
		int within_element_forwardnum = threadIdx.x - my_element_start;

		return make_int2(element_offset, within_element_forwardnum);
	}

	__device__
	static int2 assignWorkAllWarps(int* work_count, int* work_sum, int* work_offsets, SimpleScanT::TempStorage& sum_space)
	{
		int wip = threadIdx.x / 32;

		// clear work offsets
		work_offsets[wip] = 0;
		__syncthreads();

		// compute which warp should start with a given work element
		int set_offset = work_sum[threadIdx.x] - work_count[threadIdx.x];

		// set the entry for the warp to the work id
		if (work_count[threadIdx.x] > 0 && set_offset < NUM_WARPS)
			work_offsets[set_offset] = threadIdx.x;

		__syncthreads();
		// read my offset (can be the right offset or zero as only the first one will have the right per element)
		int element_offset = work_offsets[wip];

		int agg;
		SimpleScanT(sum_space). template InclusiveScan<cub::Max>(element_offset, element_offset, cub::Max(), agg);


		// every warp gets its offset and computes its offset within the element 
		int my_element_start = work_sum[element_offset] - work_count[element_offset];
		int within_element_forwardnum = wip - my_element_start;

		return make_int2(element_offset, within_element_forwardnum);
	}

public:
	__device__
	static void prepare(SharedMemT& shared_memory, SharedTempMemT& shared_temp_memory, int thread_work_count)
	{
		shared_memory.work_count[threadIdx.x] = thread_work_count;
		computeOffsets(shared_memory, shared_temp_memory, thread_work_count);
	}
	__device__
	static bool isWorkAvailable(SharedMemT& shared_memory)
	{
		volatile int* vsum = const_cast<volatile int*>(shared_memory.work_sum);
		return vsum[NUM_THREADS - 1] > 0;
	}

	__device__
	static int availableWork(SharedMemT& shared_memory)
	{
		volatile int* vsum = const_cast<volatile int*>(shared_memory.work_sum);
		return vsum[NUM_THREADS - 1];
	}

	
	__device__
	static bool pullWorkThreads(SharedMemT& shared_memory, SharedTempMemT& shared_temp_memory, int& element_offset, int& within_element_offset)
	{
		int unused;
		return pullWorkThreads(shared_memory, shared_temp_memory, element_offset, within_element_offset, unused);
	}
	__device__
	static bool pullWorkThreads(SharedMemT& shared_memory, SharedTempMemT& shared_temp_memory, int& element_offset, int& within_element_offset, int& numWork)
	{

		int2 res = assignWorkAllThreads(shared_memory.work_count, shared_memory.work_sum, shared_temp_memory.work_offsets, shared_temp_memory.tempstorage);

		element_offset = res.x;
		int within_element_forwardnum = res.y;

		// run from back to front so we can just decrese the count iif there are not enough warps for a triangle
		within_element_offset = shared_memory.work_count[element_offset] - within_element_forwardnum - 1;

		numWork = min(NUM_WARPS, shared_memory.work_sum[NUM_THREADS - 1]);

		__syncthreads();

		// update counts
		int new_work_sum = shared_memory.work_sum[threadIdx.x] - NUM_THREADS;
		shared_memory.work_count[threadIdx.x] = min(shared_memory.work_count[threadIdx.x], new_work_sum);
		shared_memory.work_sum[threadIdx.x] = new_work_sum;

		__syncthreads();
		
		// note that we might have more threads than active elements, then the within_element_offset is negative..
		return within_element_offset >= 0;
	}

	__device__
	static bool pullWorkWarps(SharedMemT& shared_memory, SharedTempMemT& shared_temp_memory, int& element_offset, int& within_element_offset)
	{
		int unused;
		return pullWorkWarps(shared_memory, shared_temp_memory, element_offset, within_element_offset, unused);
	}
	__device__
	static bool pullWorkWarps(SharedMemT& shared_memory, SharedTempMemT& shared_temp_memory, int& element_offset, int& within_element_offset, int& numWork)
	{
		int2 res = assignWorkAllWarps(shared_memory.work_count, shared_memory.work_sum, shared_temp_memory.work_offsets, shared_temp_memory.tempstorage);

		element_offset = res.x;
		int within_element_forwardnum = res.y;

		// run from back to front so we can just decrese the count iif there are not enough warps for an element
		within_element_offset = shared_memory.work_count[element_offset] - within_element_forwardnum - 1;

		numWork = min(NUM_WARPS,shared_memory.work_sum[NUM_THREADS - 1]);

		__syncthreads();

		// update counts
		int new_work_sum = shared_memory.work_sum[threadIdx.x] - NUM_WARPS;
		shared_memory.work_count[threadIdx.x] = min(shared_memory.work_count[threadIdx.x], new_work_sum);
		shared_memory.work_sum[threadIdx.x] = new_work_sum;

		__syncthreads();

		// note that we might have more threads than active elements, then the within_element_offset is negative..
		return within_element_offset >= 0;
	}

	template<typename F>
	__device__
	static bool pullWorkSelectiveThreads(SharedMemT& shared_memory, SharedTempMemT& shared_temp_memory, F f, bool deliverreversed = true)
	{
		int2 assigned_work = assignWorkAllThreads(shared_memory.work_count, shared_memory.work_sum, shared_temp_memory.work_offsets, shared_temp_memory.tempstorage);
		if (deliverreversed)
			assigned_work.y = shared_memory.work_count[assigned_work.x] - assigned_work.y - 1;

		bool res = f(shared_memory.work_count, shared_memory.work_sum, assigned_work, assigned_work.y >= 0);
		__syncthreads();
		computeOffsets(shared_memory, shared_temp_memory, shared_memory.work_count[threadIdx.x]);
		__syncthreads();
		return res;
	}
};

#endif  // INCLUDED_WORK_ASSIGNMENT
