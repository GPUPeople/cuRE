


#ifndef INCLUDED_CURE_BIN_TILE_RASTERIZATION_STAGE
#define INCLUDED_CURE_BIN_TILE_RASTERIZATION_STAGE

#pragma once

#include <math/vector.h>
#include <math/matrix.h>

#include <utils.h>
#include <warp_ops.cuh>

#include "config.h"

#include "instrumentation.cuh"

#include "BinRasterizer.cuh"
#include "TileRasterizer.cuh"
#include "StampShading.cuh"
#include "TileRasterizerMask.cuh"
#include "viewport.cuh"

#include "work_assignment.cuh"
#include "bitmask.cuh"

#include "rasterization_stage.cuh"


template <unsigned int NUM_RASTERIZERS, unsigned int NUM_WARPS, class BinTileSpace>
class BinTileRasterizationStageCommon
{
protected:
	template<unsigned int BITS_A, unsigned int BITS_B, unsigned int BITS_C>
	struct TriplePack
	{
		//static constexpr unsigned int TRI_BITS = 10U;
		//static constexpr unsigned int BIN_COORD_BITS = 11U;
		static constexpr unsigned int OFFSET_A = BITS_B + BITS_C;
		static constexpr unsigned int OFFSET_B = BITS_C;
		static constexpr unsigned int OFFSET_C = 0;
		
		static constexpr unsigned int MASK_C = (1U << BITS_C) - 1U;
		static constexpr unsigned int MASK_B = (1U << BITS_B) - 1U;
		static constexpr unsigned int MASK_BC = (1U << (BITS_B + BITS_C)) - 1U;
		unsigned int value;
	public:
		TriplePack() = default;

		__device__
			TriplePack(unsigned int a, unsigned int b, unsigned int c)
			: value((a << OFFSET_A) | (b << OFFSET_B) | (c << OFFSET_C))
		{}
		__device__ unsigned int a() const { return value >> OFFSET_A; }
		__device__ unsigned int b() const { return (value >> OFFSET_B) & MASK_B; }
		__device__ unsigned int c() const { return (value >> OFFSET_C) & MASK_C; }
		__device__ unsigned int bc() const { return value & MASK_BC; }
	};

	typedef TriplePack<10, 11, 11> BinTrianglePack;

	static_assert(NUM_WARPS <= WARP_SIZE, "Rasterization stage work assignment depends on NUM_WARPS being less than or equal WARP_SIZE");

	static constexpr int NUM_THREADS = NUM_WARPS * WARP_SIZE;

public:

	struct SharedMemT
	{
		RasterizerQueue::SortQueueShared<NUM_THREADS> rasterizer_queue_shared;
	};

	static constexpr unsigned int MAX_TRIANGLE_REFERENCES = NUM_RASTERIZERS;

	__device__
	static unsigned int enqueueTriangle(unsigned int triangle_id, unsigned int primitive_id, const math::int4& bounds)
	{
		int2 start_bin = BinTileSpace::bin(bounds.x, bounds.y);
		int2 end_bin = BinTileSpace::bin(bounds.z, bounds.w);

		return BinTileSpace::traverseRasterizers(start_bin, end_bin, [triangle_id, primitive_id](int r)
		{
			rasterizer_queue.enqueue(r, triangle_id, primitive_id);
		});
	}

	__device__
	static void writeSufficientToRunNoSync(volatile int* shared_memory)
	{
		if (threadIdx.x == 0)
			*shared_memory = rasterizer_queue.availableElements(BinTileSpace::MyQueue()) >= NUM_THREADS;
	}

	__device__
	static int fillLevelNoCheck(int qId)
	{
		return rasterizer_queue.count(qId);
	}

	__device__
	static void writeCanNotReceiveAllNoSync(volatile int* shared_memory)
	{
		if (threadIdx.x < BinTileSpace::num_rasterizers())
			if (rasterizer_queue.index_queue.size(threadIdx.x) >= static_cast<int>(RASTERIZER_QUEUE_SIZE - RASTERIZATION_CONSUME_THRESHOLD))
			{
				//printf("neglecting due to filllevel on %d: %d\n", threadIdx.x, rasterizer_queue[threadIdx.x].size());
				*shared_memory = 1;
			}
	}

	__device__
	static void writeIterateCanNotReceiveAllNoSync(volatile int* shared_memory)
	{
		for (int i = threadIdx.x; i < BinTileSpace::num_rasterizers(); i += NUM_THREADS)
			if (rasterizer_queue.index_queue.size(i) >= static_cast<int>(RASTERIZER_QUEUE_SIZE - RASTERIZATION_CONSUME_THRESHOLD))
			{
				//printf("neglecting due to filllevel on %d: %d >= %d\n", i, rasterizer_queue.index_queue.size(i), static_cast<int>(RASTERIZER_QUEUE_SIZE - RASTERIZATION_CONSUME_THRESHOLD));
				*shared_memory = 1;
			}
	}

	__device__
	static void completedPrimitive(unsigned int primitive_id)
	{
		rasterizer_queue.completedPrimitive(primitive_id);
	}

	__device__
	static bool prepareRun(char* shared_memory_in, volatile int* sufficienttorun)
	{
		bool res = rasterizer_queue.sortQueue<NUM_THREADS>(BinTileSpace::MyQueue(), shared_memory_in, sufficienttorun);
		return res;
	}
};


template <unsigned int NUM_RASTERIZERS, unsigned int NUM_WARPS, TILE_ACCESS_MODE EXCLUSIVE_TILES_RASTERMODE, bool PRIMITIVE_ORDER, bool QUAD_SHADING, class BinTileSpace, class CoverageShader, class FragmentShader, class FrameBuffer, class BlendOp>
class BinTileRasterizationStage;


template <unsigned int NUM_RASTERIZERS, unsigned int NUM_WARPS, bool PRIMITIVE_ORDER, class BinTileSpace, class CoverageShader, class FragmentShader, class FrameBuffer, class BlendOp>
class BinTileRasterizationStage<NUM_RASTERIZERS, NUM_WARPS, TILE_ACCESS_MODE::WARP_EXCLUSIVE, PRIMITIVE_ORDER, false, BinTileSpace, CoverageShader, FragmentShader, FrameBuffer, BlendOp> : public BinTileRasterizationStageCommon<NUM_RASTERIZERS, NUM_WARPS, BinTileSpace>
{
	typedef ::BinRasterizer<NUM_WARPS, BinTileSpace> BinRasterizer;
	typedef ::TileRasterizer<NUM_WARPS, true, BinTileSpace, CoverageShader, FragmentShader, FrameBuffer, BlendOp> TileRasterizer;
	typedef ::BlockWorkAssignmentOld<NUM_THREADS> BlockWorkAssignment;

	typedef ::TileBitMask<BinTileSpace> TileBitMask;
	typedef BinTileRasterizationStageCommon<NUM_RASTERIZERS, NUM_WARPS, BinTileSpace> Common;

public:
	struct SharedMemT
	{
		union
		{
			struct
			{
				unsigned int tri_ids[NUM_THREADS];
				BlockWorkAssignment::SharedMemT bin_work_assignment;
				TileBitMask tile_bit_masks[NUM_THREADS + WARP_SIZE];
				BinTrianglePack bin_triangle_pack[NUM_THREADS];
				union
				{
					BlockWorkAssignment::SharedTempMemT blockwork_sharedtemp;
					BinRasterizer::SharedMemT binraster;
					TileRasterizer::SharedMemT tileraster;
				};
				volatile unsigned int bin_work_counter[2];
			};
			Common::SharedMemT commonSMem;
		};
	};

	static constexpr size_t SHARED_MEMORY = sizeof(SharedMemT);


	__device__
	static bool run(char* shared_memory_in)
	{
		SharedMemT& shared_memory = *new(shared_memory_in) SharedMemT;

		unsigned int triidin = 0xFFFFFFFFU;
		int num_tris = rasterizer_queue.dequeueIndexBlock(BinTileSpace::MyQueue(), triidin, NUM_THREADS);
		shared_memory.tri_ids[threadIdx.x] = triidin;

		int wip = threadIdx.x / WARP_SIZE;
		if (num_tris > 0)
		{
			Instrumentation::BlockObserver<4, 1> observer;

			// clear the additional bin masks
			shared_memory.tile_bit_masks[NUM_THREADS + laneid()] = TileBitMask::Empty();

			{
				Instrumentation::BlockObserver<14, 2> observer;
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
			}

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
					math::int4 bounds;
					int2 start_bin, end_bin, binid;
					int triangleId = shared_memory.tri_ids[triangle];

					{
						Instrumentation::BlockObserver<15, 2> observer;
						// note that we could store the bin bounds in shared, but that actually does not make things faster...
						bounds = triangle_buffer.loadBounds(triangleId);
						start_bin = BinTileSpace::bin(bounds.x, bounds.y);
						end_bin = BinTileSpace::bin(bounds.z - 1, bounds.w - 1);
						binid = BinTileSpace::getHitBinForMyRasterizer(bin, start_bin, end_bin);
					}

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
						uint myBin = shared_memory.bin_triangle_pack[start + lid].bc();

						{
							Instrumentation::BlockObserver<9, 3> observer;
							#pragma unroll
							for (int i = 0; i < MaxPropagate; ++i)
							{
								TileBitMask otherMask = myMask.shfl(i);
								if (i < laneid() && shared_memory.bin_triangle_pack[start + i].bc() == myBin)
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

						BinTrianglePack& pack = shared_memory.bin_triangle_pack[start + workon];
						
						//TODO: NUM_WARPS is not right here - we need to put in the number of active waprs
						TileRasterizer::run(shared_memory.tileraster, bit, shared_memory.tri_ids[pack.a()], pack.b(), pack.c(), NUM_WARPS);
					}
					
					start = shared_memory.bin_work_counter[0];
					__syncthreads();
				}

			} while (BlockWorkAssignment::isWorkAvailable(shared_memory.bin_work_assignment));


			//////////////////////////////////////////////////////////////////////////////
			//// vis bounding box
			//__syncthreads();
			//int wip = threadIdx.x / WARP_SIZE;
			//for (int i = wip; i < num_tris; i += NUM_WARPS)
			//{
			//	math::int4 bounds = triangle_buffer.loadBounds(tri_ids[i]);
			//	int2 start_bin = BinTileSpace::bin(bounds.x, bounds.y);
			//	for (int x = bounds.x + laneid(); x < bounds.z; x += warpSize)
			//	{
			//		FrameBuffer::writeColor(x, bounds.y, make_uchar4(255, 255, 255, 255));
			//		FrameBuffer::writeColor(x, bounds.w, make_uchar4(255, 255, 255, 255));
			//	}

			//	for (int y = bounds.y + laneid(); y < bounds.w; y += warpSize)
			//	{
			//		FrameBuffer::writeColor(bounds.x, y, make_uchar4(255, 255, 255, 255));
			//		FrameBuffer::writeColor(bounds.z, y, make_uchar4(255, 255, 255, 255));
			//	}
			//}
			////////////////////////////////////////////////////////////////////////////////

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


template <unsigned int NUM_RASTERIZERS, unsigned int NUM_WARPS, bool PRIMITIVE_ORDER, class BinTileSpace, class CoverageShader, class FragmentShader, class FrameBuffer, class BlendOp>
class BinTileRasterizationStage<NUM_RASTERIZERS, NUM_WARPS, TILE_ACCESS_MODE::WARP_PER_FRAGMENT, PRIMITIVE_ORDER, false, BinTileSpace, CoverageShader, FragmentShader, FrameBuffer, BlendOp> : public BinTileRasterizationStageCommon<NUM_RASTERIZERS, NUM_WARPS, BinTileSpace>
{
	typedef ::BinRasterizer<NUM_WARPS, BinTileSpace> BinRasterizer;
	typedef ::TileRasterizer<NUM_WARPS, true, BinTileSpace, CoverageShader, FragmentShader, FrameBuffer, BlendOp> TileRasterizer;
	typedef BlockWorkAssignment<NUM_WARPS, false> BinWorkAssignment;
	typedef BlockWorkAssignment<NUM_WARPS, true>  TileWorkAssignment;

	typedef ::TileBitMask<BinTileSpace> TileBitMask;

	typedef BinTileRasterizationStageCommon<NUM_RASTERIZERS, NUM_WARPS, BinTileSpace> Common;

public:

	struct SharedMemT
	{
		union
		{
			struct
			{
				unsigned int tri_ids[NUM_THREADS];
				BinWorkAssignment::SharedMemT bin_work_assignment;
				TileWorkAssignment::SharedMemT tile_work_assignment;
				TileBitMask tile_bit_masks[NUM_THREADS];
				BinTrianglePack bin_triangle_pack[NUM_THREADS];
				union
				{
					BinWorkAssignment::SharedTempMemT bin_work_sharedtemp;
					TileWorkAssignment::SharedTempMemT tile_work_sharedtemp;
					BinRasterizer::SharedMemT binraster;
					TileRasterizer::SharedMemT tileraster;
				};
			};
			Common::SharedMemT commonSMem;
		};
	};

	static constexpr size_t SHARED_MEMORY = sizeof(SharedMemT);



	__device__
	static bool run(char* shared_memory_in)
	{
		SharedMemT& shared_memory = *new(shared_memory_in)SharedMemT;

		unsigned int triidin = 0xFFFFFFFFU;
		int num_tris = rasterizer_queue.dequeueIndexBlock(BinTileSpace::MyQueue(), triidin, NUM_THREADS);
		shared_memory.tri_ids[threadIdx.x] = triidin;

		if (num_tris > 0)
		{
			Instrumentation::BlockObserver<4, 1> observer;

			{
				Instrumentation::BlockObserver<14, 2> observer;
				int num_bins = 0;
				if (threadIdx.x < num_tris)
				{
					// compute num elements
					math::int4 bounds = triangle_buffer.loadBounds(triidin);
					int2 start_bin = BinTileSpace::bin(bounds.x, bounds.y);
					int2 end_bin = BinTileSpace::bin(bounds.z - 1, bounds.w - 1);
					num_bins = BinTileSpace::numHitBinsForMyRasterizer(start_bin, end_bin);
				}
				{
					Instrumentation::BlockObserver<7, 2> observer;
					BinWorkAssignment::prepare(shared_memory.bin_work_assignment, shared_memory.bin_work_sharedtemp, num_bins);
				}
			}

			do
			{
				__syncthreads();

				// process bin of triangle
				int triangle, bin;
				TileBitMask bitmask = TileBitMask::Empty();

				if ([&]() -> bool
				{
					Instrumentation::BlockObserver<7, 2> observer;
					return BinWorkAssignment::pullWorkThreads(shared_memory.bin_work_assignment, shared_memory.bin_work_sharedtemp, triangle, bin);
				}())
				{
					math::int4 bounds;
					int2 start_bin, end_bin, binid;
					int triangleId = shared_memory.tri_ids[triangle];

					{
						Instrumentation::BlockObserver<15, 2> observer;
						// note that we could store the bin bounds in shared, but that actually does not make things faster...
						bounds = triangle_buffer.loadBounds(triangleId);
						start_bin = BinTileSpace::bin(bounds.x, bounds.y);
						end_bin = BinTileSpace::bin(bounds.z - 1, bounds.w - 1);
						binid = BinTileSpace::getHitBinForMyRasterizer(bin, start_bin, end_bin);
					}

					// store meta information
					shared_memory.bin_triangle_pack[threadIdx.x] = BinTrianglePack(triangle, binid.x, binid.y);
					BinRasterizer::run(shared_memory.binraster, bitmask, triangleId, binid, bounds);
				}

				shared_memory.tile_bit_masks[threadIdx.x] = bitmask;
				__syncthreads();

				// work on one triangle and tile after each other (one tile per warp)
				{
					Instrumentation::BlockObserver<8, 2> observer;
					TileWorkAssignment::prepare(shared_memory.tile_work_assignment, shared_memory.tile_work_sharedtemp, bitmask.count());
					__syncthreads();
				}

				while (TileWorkAssignment::availableWork(shared_memory.tile_work_assignment) > 0)
				{ 
					{
						Instrumentation::BlockObserver<8, 2> observer;
						TileWorkAssignment::prepareConsistentWorkThreads(shared_memory.tile_work_assignment, shared_memory.tile_work_sharedtemp);
					}

					int numTiles;
					do
					{
						int tileid, tilebit;
						if ([&]() -> bool
						{
							Instrumentation::BlockObserver<8, 2> observer;
							return TileWorkAssignment::takeOutConsistentWorkThreads(warp_id(), NUM_WARPS, shared_memory.tile_work_assignment, tileid, tilebit, numTiles);
						}())
						{ 
							unsigned int bit = shared_memory.tile_bit_masks[tileid].getSetBitWarp(tilebit);
							BinTrianglePack & pack = shared_memory.bin_triangle_pack[tileid];
							TileRasterizer::run(shared_memory.tileraster, bit, shared_memory.tri_ids[pack.a()], pack.b(), pack.c(), min(NUM_WARPS, numTiles));
						}

					} while (numTiles > NUM_WARPS);

					{
						Instrumentation::BlockObserver<8, 2> observer;
						TileWorkAssignment::removeTakenWorkThreads(NUM_THREADS, shared_memory.tile_work_assignment);
					}

				}

			} while (BinWorkAssignment::availableWork(shared_memory.bin_work_assignment) > 0);


			//////////////////////////////////////////////////////////////////////////////
			//// vis bounding box
			//__syncthreads();
			//int wip = threadIdx.x / WARP_SIZE;
			//for (int i = wip; i < num_tris; i += NUM_WARPS)
			//{
			//	math::int4 bounds = triangle_buffer.loadBounds(tri_ids[i]);
			//	int2 start_bin = BinTileSpace::bin(bounds.x, bounds.y);
			//	for (int x = bounds.x + laneid(); x < bounds.z; x += warpSize)
			//	{
			//		FrameBuffer::writeColor(x, bounds.y, make_uchar4(255, 255, 255, 255));
			//		FrameBuffer::writeColor(x, bounds.w, make_uchar4(255, 255, 255, 255));
			//	}

			//	for (int y = bounds.y + laneid(); y < bounds.w; y += warpSize)
			//	{
			//		FrameBuffer::writeColor(bounds.x, y, make_uchar4(255, 255, 255, 255));
			//		FrameBuffer::writeColor(bounds.z, y, make_uchar4(255, 255, 255, 255));
			//	}
			//}
			////////////////////////////////////////////////////////////////////////////////

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


template <unsigned int NUM_RASTERIZERS, unsigned int NUM_WARPS, bool PRIMITIVE_ORDER, bool QUAD_SHADING, class BinTileSpace, class CoverageShader, class FragmentShader, class FrameBuffer, class BlendOp>
class BinTileRasterizationStage<NUM_RASTERIZERS, NUM_WARPS, TILE_ACCESS_MODE::COVERAGE_MASK, PRIMITIVE_ORDER, QUAD_SHADING, BinTileSpace, CoverageShader, FragmentShader, FrameBuffer, BlendOp> : public BinTileRasterizationStageCommon<NUM_RASTERIZERS, NUM_WARPS, BinTileSpace>
{
	typedef ::BinRasterizer<NUM_WARPS, BinTileSpace> BinRasterizer;
	typedef ::TileRasterizerMask<NUM_WARPS, BinTileSpace, CoverageShader> TileRasterizer;
	typedef ::StampShading<NUM_WARPS, TILE_RASTER_EXCLUSIVE_ACCESS_METHOD, QUAD_SHADING, BinTileSpace, CoverageShader, FragmentShader, FrameBuffer, BlendOp> StampShading;
	typedef BlockWorkAssignment<NUM_WARPS, false> BinWorkAssignment;
	typedef BlockWorkAssignment<NUM_WARPS, false>  TileWorkAssignment;
	typedef BlockWorkAssignment<NUM_WARPS, false>  ShadeWorkAssignment;

	typedef ::TileBitMask<BinTileSpace> TileBitMask;
	typedef ::StampBitMask<BinTileSpace> StampBitMask;

	typedef BinTileRasterizationStageCommon<NUM_RASTERIZERS, NUM_WARPS, BinTileSpace> Common;

public:

	typedef TriplePack<6, 10, 16> MaskTileHashPack;

	struct SharedMemT
	{
		union
		{
			struct
			{
				unsigned int tri_ids[NUM_THREADS];
				BinWorkAssignment::SharedMemT bin_work_assignment;
				TileWorkAssignment::SharedMemT tile_work_assignment;
				ShadeWorkAssignment::SharedMemT shade_work_assignment;
				TileBitMask tile_bit_masks[NUM_THREADS];
				BinTrianglePack bin_triangle_pack[NUM_THREADS];
				StampBitMask stamp_bit_masks[NUM_THREADS];
				MaskTileHashPack tile_bin_mask_pack[NUM_THREADS];
				union
				{
					BinWorkAssignment::SharedTempMemT bin_work_sharedtemp;
					TileWorkAssignment::SharedTempMemT tile_work_sharedtemp;
					ShadeWorkAssignment::SharedTempMemT shade_work_sharedtemp;
					BinRasterizer::SharedMemT binraster;
					TileRasterizer::SharedMemT tileraster;
					StampShading::SharedMemT stampshading;
				};
			};
			Common::SharedMemT commonSMem;
		};
	};

	static constexpr size_t SHARED_MEMORY = sizeof(SharedMemT);



	__device__
	static bool run(char* shared_memory_in)
	{
		SharedMemT& shared_memory = *new(shared_memory_in)SharedMemT;

		unsigned int triidin = 0xFFFFFFFFU;
		int num_tris = rasterizer_queue.dequeueIndexBlock(BinTileSpace::MyQueue(), triidin, NUM_THREADS);
		shared_memory.tri_ids[threadIdx.x] = triidin;

		if (num_tris > 0)
		{
			Instrumentation::BlockObserver<4, 1> observer;

			{
				Instrumentation::BlockObserver<14, 2> observer;
				int num_bins = 0;
				if (threadIdx.x < num_tris)
				{
					// compute num elements
					math::int4 bounds = triangle_buffer.loadBounds(triidin);
					int2 start_bin = BinTileSpace::bin(bounds.x, bounds.y);
					int2 end_bin = BinTileSpace::bin(bounds.z - 1, bounds.w - 1);
					num_bins = BinTileSpace::numHitBinsForMyRasterizer(start_bin, end_bin);
				}
				{
					Instrumentation::BlockObserver<7, 2> observer;
					BinWorkAssignment::prepare(shared_memory.bin_work_assignment, shared_memory.bin_work_sharedtemp, num_bins);
				}
			}

			do
			{
				__syncthreads();

				// process bin of triangle
				int triangle, bin;
				TileBitMask bitmask = TileBitMask::Empty();

				if ([&]() -> bool
				{
					Instrumentation::BlockObserver<7, 2> observer;
					return BinWorkAssignment::pullWorkThreads(shared_memory.bin_work_assignment, shared_memory.bin_work_sharedtemp, triangle, bin);
				}())
				{
					math::int4 bounds;
					int2 start_bin, end_bin, binid;
					int triangleId = shared_memory.tri_ids[triangle];

					{
						Instrumentation::BlockObserver<15, 2> observer;
						// note that we could store the bin bounds in shared, but that actually does not make things faster...
						bounds = triangle_buffer.loadBounds(triangleId);
						start_bin = BinTileSpace::bin(bounds.x, bounds.y);
						end_bin = BinTileSpace::bin(bounds.z - 1, bounds.w - 1);
						binid = BinTileSpace::getHitBinForMyRasterizer(bin, start_bin, end_bin);
					}

					// store meta information
					shared_memory.bin_triangle_pack[threadIdx.x] = BinTrianglePack(triangle, binid.x, binid.y);
					BinRasterizer::run(shared_memory.binraster, bitmask, triangleId, binid, bounds);

				/*	for (int r = 0; r < 8; ++r)
						for (int c = 0; c < 8; ++c)
						{
							if (bitmask.isset(c, r))
							{
								int2 localtile = make_int2(c, r);
								int4 tbound = BinTileSpace::tileBounds(binid, localtile);
								for (int y = tbound.y; y < tbound.w; ++y)
									for (int x = tbound.x; x < tbound.z; ++x)
										FrameBuffer::writeColor(x, y, make_uchar4(triangleId % 256, triangleId / 256 % 256, triangleId / 256 / 256 % 256, 255));
							}
						}*/
				}

				shared_memory.tile_bit_masks[threadIdx.x] = bitmask;
				__syncthreads();


				{
					Instrumentation::BlockObserver<8, 2> observer;
					TileWorkAssignment::prepare(shared_memory.tile_work_assignment, shared_memory.tile_work_sharedtemp, bitmask.count());
					//__syncthreads();
				}

				do
				{
					__syncthreads();
					int tile, tilebit;
					StampBitMask stampbitmask = StampBitMask::Empty();

					if ([&]() -> bool
					{
						Instrumentation::BlockObserver<8, 2> observer;
						return TileWorkAssignment::pullWorkThreads(shared_memory.tile_work_assignment, shared_memory.tile_work_sharedtemp, tile, tilebit);
					}())
					{
						BinTrianglePack & pack = shared_memory.bin_triangle_pack[tile];

						unsigned int bit = shared_memory.tile_bit_masks[tile].getSetBit(tilebit);
						int2 localTile = TileBitMask::bitToCoord(bit);
						int2 bin = make_int2(pack.b(), pack.c());
						int triangleId = shared_memory.tri_ids[pack.a()];
						shared_memory.tile_bin_mask_pack[threadIdx.x] = MaskTileHashPack(bit, tile, ((bin.x << 6) ^ (bin.y << 3) ^ bit) & 0xFFFF);
						TileRasterizer::run(shared_memory.tileraster, stampbitmask, triangleId, bin, localTile);

						//int4 tbound = BinTileSpace::tileBounds(bin, localTile);
						//for (int y = 0; y < 8; ++y)
						//	for (int x = 0; x < 8; ++x)
						//		if (stampbitmask.isset(x, y))
						//		{ 
						//			FrameBuffer::writeColor(tbound.x + x, tbound.y + y, make_uchar4(triangleId % 256, triangleId / 256 % 256, triangleId / 256 / 256 % 256, 255));
						//		}
					}
					shared_memory.stamp_bit_masks[threadIdx.x] = stampbitmask;

					__syncthreads();

					{
						Instrumentation::BlockObserver<17, 2> observer;
						ShadeWorkAssignment::prepare(shared_memory.shade_work_assignment, shared_memory.shade_work_sharedtemp, QUAD_SHADING ? (stampbitmask.quadMask().count()*4) : stampbitmask.count());
						__syncthreads();
					}

					while (ShadeWorkAssignment::availableWork(shared_memory.shade_work_assignment) > 0)
					{
						__syncthreads();
						int stampmaskoffset = 0, localoffset = 0, stampbit, sumwork, startTile;
						int triangleId, bit;
						int2 p;
						if ([&]() -> bool
						{
							Instrumentation::BlockObserver<17, 2> observer;
							return ShadeWorkAssignment::pullWorkThreads(shared_memory.shade_work_assignment, shared_memory.shade_work_sharedtemp, stampmaskoffset, stampbit, sumwork, startTile, localoffset);
						}())
						{

							MaskTileHashPack & tilepack = shared_memory.tile_bin_mask_pack[stampmaskoffset];
							BinTrianglePack & binpack = shared_memory.bin_triangle_pack[tilepack.b()];

							StampBitMask& myBitMask = shared_memory.stamp_bit_masks[stampmaskoffset];
							
							bit = QUAD_SHADING ? myBitMask.quadMask().getSetBit(stampbit/4) : myBitMask.getSetBit(stampbit);
							bit = QUAD_SHADING ? (bit + threadIdx.x % 2 + (threadIdx.x % 4 / 2) * StampBitMask::Cols) : bit;
							int2 localStamp = StampBitMask::bitToCoord(bit);
							int2 localTile = TileBitMask::bitToCoord(tilepack.a());
							int2 bin = make_int2(binpack.b(), binpack.c());
							triangleId = shared_memory.tri_ids[binpack.a()];

							//run shading and blending

							int4 tbound = BinTileSpace::tileBounds(bin, localTile);
							p = make_int2(tbound.x + localStamp.x, tbound.y + localStamp.y);

							
							//FrameBuffer::writeColor(p.x, p.y, make_uchar4(triangleId % 256, triangleId / 256 % 256, triangleId / 256 / 256 % 256, 255));
						}

						//__syncthreads();

						//TODO: we need to use the compacted bin and the tileid within the bin for the comparison
						StampShading::run(shared_memory.stampshading, triangleId, p, shared_memory.stamp_bit_masks, stampmaskoffset, stampbit, bit, localoffset, startTile, sumwork, [&shared_memory](int i){return shared_memory.tile_bin_mask_pack[i].c(); });

						//__syncthreads();

					}

				} while (TileWorkAssignment::availableWork(shared_memory.tile_work_assignment) > 0);

			} while (BinWorkAssignment::availableWork(shared_memory.bin_work_assignment) > 0);


			//////////////////////////////////////////////////////////////////////////////
			//// vis bounding box
			//__syncthreads();
			//int wip = threadIdx.x / WARP_SIZE;
			//for (int i = wip; i < num_tris; i += NUM_WARPS)
			//{
			//	math::int4 bounds = triangle_buffer.loadBounds(tri_ids[i]);
			//	int2 start_bin = BinTileSpace::bin(bounds.x, bounds.y);
			//	for (int x = bounds.x + laneid(); x < bounds.z; x += warpSize)
			//	{
			//		FrameBuffer::writeColor(x, bounds.y, make_uchar4(255, 255, 255, 255));
			//		FrameBuffer::writeColor(x, bounds.w, make_uchar4(255, 255, 255, 255));
			//	}

			//	for (int y = bounds.y + laneid(); y < bounds.w; y += warpSize)
			//	{
			//		FrameBuffer::writeColor(bounds.x, y, make_uchar4(255, 255, 255, 255));
			//		FrameBuffer::writeColor(bounds.z, y, make_uchar4(255, 255, 255, 255));
			//	}
			//}
			////////////////////////////////////////////////////////////////////////////////

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

#endif  // INCLUDED_CURE_BIN_TILE_RASTERIZATION_STAGE
