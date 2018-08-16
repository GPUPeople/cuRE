


#ifndef INCLUDED_CURE_TRIANGLE_BLOCK_BIN_TILE_WORK_ASSIGNMENT
#define INCLUDED_CURE_TRIANGLE_BLOCK_BIN_TILE_WORK_ASSIGNMENT

#pragma once

#include <math/vector.h>
#include <math/matrix.h>

#include <utils.h>
#include <warp_ops.cuh>

#include "config.h"

#include "instrumentation.cuh"

#include "BinRasterizer.cuh"
#include "TileRasterizer.cuh"
#include "viewport.cuh"

#include "work_assignment.cuh"
#include "bitmask.cuh"

#include "rasterization_stage.cuh"



template <unsigned int NUM_BLOCKS, unsigned int NUM_WARPS, class BinRasterizer>
class TriangleBlockBinTileWorkAssignment
{
private:
	static_assert(NUM_WARPS <= WARP_SIZE, "Rasterization stage work assignment depends on NUM_WARPS being less than or equal WARP_SIZE");

	static constexpr int NUM_THREADS = NUM_WARPS * WARP_SIZE;

	typedef ::BlockWorkAssignment<NUM_THREADS> BlockWorkAssignment;

	struct BinTrianglePack
	{
		static constexpr unsigned int TRI_BITS = 10U;
		static constexpr unsigned int BIN_COORD_BITS = 11U;
		static constexpr unsigned int TRI_OFFSET = 2U * BIN_COORD_BITS;
		static constexpr unsigned int BIN_X_OFFSET = 0;
		static constexpr unsigned int BIN_Y_OFFSET = BIN_COORD_BITS;
		static constexpr unsigned int BIN_MASK = (1U << BIN_COORD_BITS) - 1U;
		static constexpr unsigned int COMBBIN_MASK = (1U << (2 * BIN_COORD_BITS)) - 1U;
		unsigned int value;
	public:
		__device__ BinTrianglePack() = default;

		__device__
		BinTrianglePack(unsigned int localTriangleId, unsigned int bin_x, unsigned int bin_y)
			: value((localTriangleId << TRI_OFFSET) | (bin_x << BIN_X_OFFSET) | (bin_y << BIN_Y_OFFSET))
		{}
		__device__ unsigned int triId() const { return value >> TRI_OFFSET; }
		__device__ unsigned int binX() const { return (value >> BIN_X_OFFSET) & BIN_MASK; }
		__device__ unsigned int binY() const { return (value >> BIN_Y_OFFSET) & BIN_MASK; }
		__device__ unsigned int combbin() const { return value & COMBBIN_MASK; }
	};


public:

	struct SharedMemT
	{
		unsigned int tri_ids[NUM_THREADS];
		BlockWorkAssignment::SharedMemT bin_work_assignment;
		TileBitMask tile_bit_masks[NUM_THREADS + WARP_SIZE];
		BinTrianglePack bin_triangle_pack[NUM_THREADS];
		union
		{
			BlockWorkAssignment::SharedTempMemT blockwork_sharedtemp;
			BinRasterizer::SharedMemT binraster;
		};
		volatile unsigned int bin_work_counter[2];
	};

	static constexpr size_t SHARED_MEMORY = sizeof(SharedMemT);

	__device__
	static void writeSufficientToRunNoSync(volatile int* shared_memory)
	{
		if (threadIdx.x == 0)
			*shared_memory = rasterizer_queue.size(BinTileSpace::MyQueue()) >= NUM_THREADS;
	}

	__device__
	static void writeCanNotReceiveAllNoSync(volatile int* shared_memory)
	{
		if (threadIdx.x < BinTileSpace::num_rasterizers())
			if (rasterizer_queue.size(threadIdx.x) >= static_cast<int>(RASTERIZER_QUEUE_SIZE - RASTERIZATION_CONSUME_THRESHOLD))
			{
				//printf("neglecting due to filllevel on %d: %d\n", threadIdx.x, rasterizer_queue[threadIdx.x].size());
				*shared_memory = 1;
			}
	}

	__device__
	static bool run(char* shared_memory_in)
	{
		SharedMemT& shared_memory = *new(shared_memory_in) SharedMemT;

		unsigned int triidin = 0xFFFFFFFFU;
		int num_tris = rasterizer_queue.dequeueBlock(BinTileSpace::MyQueue(), &triidin, NUM_THREADS);
		shared_memory.tri_ids[threadIdx.x] = triidin;

		int wip = threadIdx.x / WARP_SIZE;
		if (num_tris > 0)
		{
			Instrumentation::BlockObserver<4, 1> observer;

			// clear the additional bin masks
			shared_memory.tile_bit_masks[NUM_THREADS + laneid()] = TileBitMask::Empty();

			int num_bins = 0;
			if (threadIdx.x < num_tris)
			{
				// compute num elements
				math::int4 bounds = triangle_buffer.loadBounds(triidin);
				int2 start_bin = BinTileSpace::bin(bounds.x, bounds.y);
				int2 end_bin = BinTileSpace::bin(bounds.z - 1, bounds.w - 1);
				num_bins = BinTileSpace::numHitBinsForMyRasterizer(start_bin, end_bin);
			}
			BlockWorkAssignment::prepare(shared_memory.bin_work_assignment, shared_memory.blockwork_sharedtemp, num_bins);
			
			do
			{
				__syncthreads();

				// process bin of triangle
				int triangle, bin;
				TileBitMask bitmask = TileBitMask::Empty();

				// store numbins in shared so we dont waste a register
				shared_memory.bin_work_counter[1] = min(NUM_THREADS, BlockWorkAssignment::availableWork(shared_memory.bin_work_assignment));

				if ([&]() -> bool
				{
					Instrumentation::BlockObserver<7, 2> observer;
					return BlockWorkAssignment::pullWorkThreads(shared_memory.bin_work_assignment, shared_memory.blockwork_sharedtemp, triangle, bin);
				}())
				{
					int triangleId = shared_memory.tri_ids[triangle];

					// note that we could store the bin bounds in shared, but that actually does not make things faster...
					math::int4 bounds = triangle_buffer.loadBounds(triangleId);
					int2 start_bin = BinTileSpace::bin(bounds.x, bounds.y);
					int2 end_bin = BinTileSpace::bin(bounds.z - 1, bounds.w - 1);
					int2 binid = BinTileSpace::getHitBinForMyRasterizer(bin, start_bin, end_bin);

					// store meta information
					shared_memory.bin_triangle_pack[threadIdx.x] = BinTrianglePack(triangle, binid.x, binid.y);
					BinRasterizer::run(shared_memory.binraster, bitmask, triangleId, binid, bounds);
					//if (blockIdx.x == 16)
					//	printf("%d: %llx\n", threadIdx.x, bitmask.mask);
				}

				shared_memory.tile_bit_masks[threadIdx.x] = bitmask;
				__syncthreads();

				// every warp figures out if there are other warps working on the same tile
				// updates its internal bit mask copies
				// computes the work offsets
				// and picks its own tile

				int start = 0;
				int lid = laneid();
				while (start < shared_memory.bin_work_counter[1])
				{
					TileBitMask myMask;
					unsigned int c, workon, mCounter;

					{
						Instrumentation::BlockObserver<8, 2> observer;

						const unsigned int MaxPropagate = 15U;
						myMask = shared_memory.tile_bit_masks[start + lid];
						uint myBin = shared_memory.bin_triangle_pack[start + lid].combbin();

						{
							Instrumentation::BlockObserver<9, 3> observer;
							#pragma unroll
							for (int i = 0; i < MaxPropagate; ++i)
							{
								TileBitMask otherMask = myMask.shfl(i);
								if (i < laneid() && shared_memory.bin_triangle_pack[start + i].combbin() == myBin)
									myMask.unmark(otherMask);
							}
						}


						mCounter = myMask.count();
						if (MaxPropagate < 31U && lid > MaxPropagate)
							mCounter = 0;

						WarpScan<unsigned int>::InclusiveSum(mCounter, c, lid);
						workon = __ffs(__ballot_sync(~0U, c > wip)) - 1;


						if (wip == NUM_WARPS - 1)
						{
							bool changed = !(myMask == shared_memory.tile_bit_masks[start + lid]);
							unsigned int nchanged = __ffs(__ballot_sync(~0U, changed)) - 1;
							unsigned int nextstart = start + min(MaxPropagate + 1, min(nchanged, workon));
							shared_memory.bin_work_counter[0] = nextstart;
						}

						__syncthreads();
					}

					if (workon != 0xFFFFFFFFU)
					{
						// distribute offset and bitmask
						unsigned int bitoffsetid = wip + mCounter - c;
						bitoffsetid = __shfl_sync(~0U, bitoffsetid, workon);
						myMask = myMask.shfl(workon);

						// find the bit
						unsigned int bit = myMask.getSetBitWarp(bitoffsetid);

						// if i am the last one to work on that bitmask, unset all of us
						if (lid == workon && (wip == NUM_WARPS - 1 || bitoffsetid + 1 == mCounter))
						{
							myMask.andStride(0, bit + 1);
							shared_memory.tile_bit_masks[start + lid].unmark(myMask);
						}

						BinTrianglePack & pack = shared_memory.bin_triangle_pack[start + workon];
						
						TileRasterizer::run(shared_memory.tileraster, bit, shared_memory.tri_ids[pack.triId()], pack.binX(), pack.binY());
					}
					
					start = shared_memory.bin_work_counter[0];
					__syncthreads();
				}

			} while (BlockWorkAssignment::isWorkAvailable(shared_memory.bin_work_assignment));

			__threadfence();
			if (shared_memory.tri_ids[threadIdx.x] != 0xFFFFFFFFU)
			{
				triangle_buffer.release(shared_memory.tri_ids[threadIdx.x]);
			}
			return true;
		}

		return false;
	}
};

#endif  // INCLUDED_CURE_TRIANGLE_BLOCK_BIN_TILE_WORK_ASSIGNMENT
