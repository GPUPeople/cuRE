#ifndef INCLUDED_CURE_RASTERIZER2
#define INCLUDED_CURE_RASTERIZER2

#pragma once



extern "C"
{
	__device__ unsigned int debugCounters[1024];
}

#include <math/vector.h>
#include <math/matrix.h>

#include "triangle_buffer.cuh"
#include "index_queue.cuh"
#include "bin_raster.cuh"
#include "tile_raster.cuh"
#include "bitmask.cuh"

#include "config.h"
#include "utils.cuh"


#include "viewport.cuh"


#include "utils/work_assignment.cuh"

#include "buffer_definitions.cuh"



template <unsigned int NUM_WARPS, class RasterizerSpace, class CoverageShader, class FragmentShader, class FrameBuffer>
class RasterizationStage
{
private:
	typedef typename ::FragmentShaderInputSignature<decltype(FragmentShader::shade)>::type FragmentShaderInputSignature;

	static_assert(NUM_WARPS <= WARP_SIZE, "NUM_WARPS must be smaller equal WARP_SIZE: Rasterization stage work assignment depends on decissions made in a single warp");

	static constexpr int NUM_THREADS = NUM_WARPS*WARP_SIZE;

	typedef ::BinRasterizer<NUM_WARPS, RasterizerSpace> BinRasterizer;
	typedef ::TileRasterizer<NUM_WARPS, RasterizerSpace, CoverageShader, FragmentShader, FrameBuffer> TileRasterizer;
	typedef ::BlockWorkAssignment<NUM_THREADS> BlockWorkAssignment;

	typedef ::TileBitMask<RasterizerSpace> TileBitMask;

	struct BinTrianglePack
	{
		static constexpr unsigned int TRI_BITS = 10U;
		static constexpr unsigned int BIN_COORD_BITS = 11U;
		static constexpr unsigned int TRI_OFFSET = 2U * BIN_COORD_BITS;
		static constexpr unsigned int BIN_X_OFFSET = 0;
		static constexpr unsigned int BIN_Y_OFFSET = BIN_COORD_BITS;
		static constexpr unsigned int BIN_MASK = (1U << BIN_COORD_BITS) - 1U;
		static constexpr unsigned int COMBBIN_MASK = (1U << (2*BIN_COORD_BITS))-1U;
		unsigned int value;
	public:
		__device__
		BinTrianglePack(unsigned int localTriangleId, unsigned int bin_x, unsigned int bin_y) 
			: value((localTriangleId << TRI_OFFSET) | (bin_x << BIN_X_OFFSET) | (bin_y << BIN_Y_OFFSET))
		{ }
		__device__ unsigned int triId() const { return value >> TRI_OFFSET; }
		__device__ unsigned int  binX() const { return (value >> BIN_X_OFFSET) & BIN_MASK; }
		__device__ unsigned int  binY() const { return (value >> BIN_Y_OFFSET) & BIN_MASK; }
		__device__ unsigned int  combbin() const { return value & COMBBIN_MASK; }
	};

public:
	static constexpr size_t SHARED_MEMORY = sizeof(unsigned int)*NUM_THREADS
		+ 2 * BlockWorkAssignment::SHARED_MEMORY
		+ sizeof(TileBitMask)* NUM_THREADS
		+ sizeof(BinTrianglePack) * NUM_THREADS
		+ NUM_THREADS*sizeof(ushort2)
		+ static_max<BlockWorkAssignment::SHARED_TEMP_MEMORY, BinRasterizer::SHARED_MEMORY, TileRasterizer::SHARED_MEMORY>::value;


	__device__
		static unsigned int enqueueTriangle(unsigned int triangle_id, const math::int4& bounds)
	{
		int2 start_bin = RasterizerSpace::bin(bounds.x, bounds.y);
		int2 end_bin = RasterizerSpace::bin(bounds.z, bounds.w);

		return RasterizerSpace::traverseRasterizers(start_bin, end_bin, [triangle_id](int r)
		{
			rasterizer_queue[r].enqueue(triangle_id);
		});
	}

	__device__
	static bool sufficientToRun(char* shared_memory)
	{
		int* num = reinterpret_cast<int*>(shared_memory + SHARED_MEMORY - 3 * sizeof(int));
		if (threadIdx.x == 0)
			*num = rasterizer_queue[RasterizerSpace::MyQueue()].size();
		__syncthreads();
		return *num >= NUM_THREADS;
	}

	__device__
	static bool run(char* shared_memory)
	{

		unsigned int* tri_ids = reinterpret_cast<unsigned int*>(shared_memory);
		char* bin_work_assignment_shared = shared_memory + sizeof(unsigned int)*NUM_THREADS;
		char* tile_work_assignment_shared = bin_work_assignment_shared + BlockWorkAssignment::SHARED_MEMORY;
		char* c_tile_bit_masks = tile_work_assignment_shared + BlockWorkAssignment::SHARED_MEMORY;
		TileBitMask* tile_bit_masks = reinterpret_cast<TileBitMask*>(c_tile_bit_masks);
		char* c_bin_triangle_pack = c_tile_bit_masks + sizeof(TileBitMask)* NUM_THREADS;
		BinTrianglePack* bin_triangle_pack = reinterpret_cast<BinTrianglePack*>(c_bin_triangle_pack);
		char* warp_work_assignment = c_bin_triangle_pack + sizeof(BinTrianglePack)* NUM_THREADS;
		char* shared_temp = warp_work_assignment + sizeof(ushort2)* NUM_THREADS;
		char* tile_raster_shared = shared_temp;

		unsigned int triidin = 0xFFFFFFFFU;
		int num_tris = rasterizer_queue[RasterizerSpace::MyQueue()].dequeueBlock(&triidin, NUM_THREADS);
		tri_ids[threadIdx.x] = triidin;

		if (num_tris > 0)
		{
			int num_bins = 0;
			if (threadIdx.x < num_tris)
			{
				// compute num elements
				math::int4 bounds = triangle_buffer.loadBounds(triidin);
				int2 start_bin = RasterizerSpace::bin(bounds.x, bounds.y);
				int2 end_bin = RasterizerSpace::bin(bounds.z, bounds.w);
				num_bins = RasterizerSpace::numHitBinsForMyRasterizer(start_bin, end_bin);
			}
			BlockWorkAssignment::prepare(bin_work_assignment_shared, shared_temp, num_bins);
			__syncthreads();

			do
			{
				// process bin of triangle
				int triangle, bin;
				int num_tiles = 0;

				if (BlockWorkAssignment::pullWorkThreads(bin_work_assignment_shared, shared_temp, triangle, bin))
				{
					//if (blockIdx.x == 2)
					//	printf("%d got %d %d\n", threadIdx.x, triangle, bin);
					int triangleId = tri_ids[triangle];
					math::int4 bounds = triangle_buffer.loadBounds(triangleId);
					int2 start_bin = RasterizerSpace::bin(bounds.x, bounds.y);
					int2 end_bin = RasterizerSpace::bin(bounds.z, bounds.w);
					int2 binid = RasterizerSpace::getHitBinForMyRasterizer(bin, start_bin, end_bin);
					// store meta information
					bin_triangle_pack[threadIdx.x] = BinTrianglePack(triangle, binid.x, binid.y);
					num_tiles = BinRasterizer::run(shared_temp, tile_bit_masks, triangleId, binid);
				}
				__syncthreads();

				// assign tiles
				BlockWorkAssignment::prepare(tile_work_assignment_shared, shared_temp, num_tiles);
				__syncthreads();
				do
				{
					int wip = threadIdx.x / WARP_SIZE;
					ushort2 *warpdata = reinterpret_cast<ushort2*>(warp_work_assignment);


					// process tile of triangle
					BlockWorkAssignment::pullWorkSelectiveThreads(tile_work_assignment_shared, shared_temp,
						[&tile_raster_shared, &tri_ids, &warpdata, &tile_bit_masks, &bin_triangle_pack, wip](int* count, int* sum_count, int2 threadWork, bool haswork)->bool
					{
						// write work assignment to shared
						// TODO: make sure different warps are not working on the same tile in parallel!

						//// one warp makes sure that we do not end up with different triangles for the same tile
						//if (threadIdx.x < 32)
						//{
						//	unsigned int mytri = bin_triangle_pack[threadWork.x].triId();
						//	unsigned int combbin = bin_triangle_pack[threadWork.x].combbin();
						//	bool canwork = true;
						//	#pragma unroll
						//	for (int i = 0; i < 32; ++i)
						//	{
						//		unsigned int vtre = __shfl_sync(~0U, mytri, i);
						//		unsigned int vcombbin = __shfl_sync(~0U, combbin, i);
						//		TileBitMask vBinmask = tile_bit_masks[threadWork.x].shfl(i);
						//		if (threadIdx.x > i && vtre != mytri && vcombbin == combbin && vBinmask.overlap(tile_bit_masks[threadWork.x]))
						//			canwork = false;
						//	}
						//	unsigned int workmask = __ballot_sync(~0U, canwork);
						//	unsigned int numwork = min(NUM_WARPS, __popc(workmask));
						//	int myworkoffset = __popc(workmask & lanemask_lt());
						//	if (canwork && myworkoffset < numwork)
						//	{ 
						//		warpdata[myworkoffset] = make_int2(threadWork.x, count[threadWork.x] - threadWork.y - 1);
						//		if (__shfl_down_sync(~0U, threadWork.x, 1) != threadWork.x || (myworkoffset + 1 == numwork))
						//		{
						//			count[threadWork.x] = max(0, count[threadWork.x] - threadWork.y - 1);
						//		}
						//	}
						//	if (threadIdx.x >= numwork && threadIdx.x < NUM_WARPS)
						//		warpdata[threadIdx.x] = make_int2(0, -1);
						//}
						
						
						warpdata[threadIdx.x] = make_ushort2(threadWork.x, threadWork.y >= 0 ? threadWork.y : 0xFFFFU);
						__syncthreads();

						#pragma unroll
						for (int i = 0; i < NUM_THREADS; i += NUM_WARPS)
						{ 
							
							uint2 tw = make_uint2(warpdata[i + wip].x, warpdata[i + wip].y);
							if (tw.y != 0xFFFFU)
							{
								int tileid = tile_bit_masks[tw.x].getSetBitWarp(tw.y);
								TileRasterizer::run(tile_raster_shared, tileid,
									tri_ids[bin_triangle_pack[tw.x].triId()],
									bin_triangle_pack[tw.x].binX(), bin_triangle_pack[tw.x].binY());
							}
							
						}

						__syncthreads();
						//// for now just take next best tile and reduce count
						//if (threadIdx.x < NUM_WARPS)
						//	warpdata[threadIdx.x] = threadWork;
						//	//warpdata[threadIdx.x] = make_int2(threadWork.y>=0?threadWork.x:-1, tile_bit_masks[threadWork.x].getSetBit(threadWork.y));
						//count[threadIdx.x] = max(0, min(count[threadIdx.x], sum_count[threadIdx.x] - static_cast<int>(NUM_WARPS)));

						count[threadIdx.x] = max(0, min(count[threadIdx.x], sum_count[threadIdx.x] - static_cast<int>(NUM_THREADS)));

						return true;
					}, true);

					
				} while (BlockWorkAssignment::isWorkAvailable(tile_work_assignment_shared));

			} while (BlockWorkAssignment::isWorkAvailable(bin_work_assignment_shared));


			//////////////////////////////////////////////////////////////////////////////
			//// vis bounding box
			//__syncthreads();
			//int wip = threadIdx.x / WARP_SIZE;
			//for (int i = wip; i < num_tris; i += NUM_WARPS)
			//{
			//	math::int4 bounds = triangle_buffer.loadBounds(tri_ids[i]);
			//	int2 start_bin = RasterizerSpace::bin(bounds.x, bounds.y);
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
			if (tri_ids[threadIdx.x] != 0xFFFFFFFFU)
			{
				triangle_buffer.release(tri_ids[threadIdx.x]);
			}
			return true;
		}

		return false;
	}
};

#endif
