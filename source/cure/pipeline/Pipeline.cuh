


#ifndef INCLUDED_CURE_PIPELINE
#define INCLUDED_CURE_PIPELINE

#pragma once

#include "PerWarpPatchGeometryStage.cuh"
#include "BinTileSpace.cuh"
#include "BinTileRasterizationStage.cuh"
#include "framebuffer.cuh"

#include "shader.cuh"

#include <meta_utils.h>
#include <ptx_primitives.cuh>
#include <cub/cub.cuh>

extern "C"
{
	__device__ int geometryProducingBlocksCount;
}

template <bool ENABLE, unsigned int NUM_RASTERIZERS, unsigned int NUM_WARPS, unsigned int DYNAMIC_RASTERIZERS, class InputVertexAttributes, class PrimitiveType, class VertexShader, class CoverageShader, class FragmentShader, class BlendOp>
class Pipeline;

template <unsigned int NUM_RASTERIZERS, unsigned int NUM_WARPS, unsigned int DYNAMIC_RASTERIZERS, class InputVertexAttributes, class PrimitiveType, class VertexShader, class CoverageShader, class FragmentShader, class BlendOp>
class Pipeline<false, NUM_RASTERIZERS, NUM_WARPS, DYNAMIC_RASTERIZERS, InputVertexAttributes, PrimitiveType, VertexShader, CoverageShader, FragmentShader, BlendOp>
{
public:
	__device__
	static void run()
	{
		if (blockIdx.x == 0 && threadIdx.x == 0)
			printf("WARNING: pipeline disabled\n");
	}
};

template <unsigned int NUM_RASTERIZERS, unsigned int NUM_WARPS, class InputVertexAttributes, class PrimitiveType, class VertexShader, class CoverageShader, class FragmentShader, class BlendOp>
class Pipeline<true, NUM_RASTERIZERS, NUM_WARPS, 0, InputVertexAttributes, PrimitiveType, VertexShader, CoverageShader, FragmentShader, BlendOp>
{
	typedef PatternTileSpace<PATTERN_TECHNIQUE, NUM_RASTERIZERS, 8192, 8192, 8, 8, 8, 8> UsedBinTileSpace;
	typedef BinTileRasterizationStage<NUM_RASTERIZERS, NUM_WARPS, BINRASTER_EXCLUSIVE_TILE_ACCESS_MODE, ENFORCE_PRIMITIVE_ORDER, FORCE_QUAD_SHADING, UsedBinTileSpace, CoverageShader, FragmentShader, FrameBuffer, BlendOp> RasterizationStage;
	typedef PerWarpPatchCachedGeometryStage<NUM_WARPS, InputVertexAttributes, PrimitiveType, VertexShader, FragmentShader, RasterizationStage, CLIPPING> GeometryStage;

	struct SharedState
	{
		union
		{
			__align__(16) char rasterization_stage_shared_memory[RasterizationStage::SHARED_MEMORY];
			__align__(16) char geometry_stage_shared_memory[GeometryStage::SHARED_MEMORY];
		};
	};

public:
	__device__
	static void run()
	{
		__shared__ SharedState shared_memory;

		Instrumentation::BlockObserver<0, 0> observer;

		__shared__ volatile int runstate[6];

		//runstate:
		// 0 geometry active
		// 1 rasterizer efficient
		// 2 rasterizer full
		// 3 require sortrun
		// 4 require rasterrun
		// 5 other geom running

		runstate[0] = true;
		runstate[1] = false;
		runstate[2] = false;
		runstate[5] = true;
		__syncthreads();

		while (runstate[3] || runstate[4] || runstate[5])
		{
			if (runstate[0] && !runstate[1])
			{
				RasterizationStage::writeCanNotReceiveAllNoSync(&runstate[2]);
				__syncthreads();
				if (!runstate[2])
				{
					if (!GeometryStage::run(shared_memory.geometry_stage_shared_memory))
					{
						if (threadIdx.x == 0)
						{
							atomicSub(&geometryProducingBlocksCount, 1);
							runstate[0] = false;
						}
					}
				}
			}
			__syncthreads();

			runstate[3] = RasterizationStage::prepareRun(shared_memory.rasterization_stage_shared_memory, &runstate[1]);
			__syncthreads();
			if (!runstate[0] || runstate[1] || runstate[2])
			{
				runstate[4] = RasterizationStage::run(shared_memory.rasterization_stage_shared_memory);
				runstate[2] = false;
			}
			runstate[5] = ldg_cg(&geometryProducingBlocksCount) != 0;
			__syncthreads();
		}
	}
};


template<int PER_THREAD_CHECK, int PER_THREAD_SORT, unsigned int DYNAMIC_RASTERIZERS, unsigned int NUM_THREADS, bool PRESORT>
struct PipelineSortingElements
{
	static_assert(PER_THREAD_CHECK <= PER_THREAD_SORT, "when PRESORT is false, PER_THREAD_CHECK <= PER_THREAD_SORT must hold");
	static constexpr int SORTING_ELEMENTS = PER_THREAD_CHECK;

	template<typename F>
	__device__
	static void loadin(unsigned int(&counts)[SORTING_ELEMENTS], int(&ids)[SORTING_ELEMENTS], F f)
	{
		#pragma unroll
		for (int i = 0; i < SORTING_ELEMENTS; i++)
		{
			int id = i * NUM_THREADS + threadIdx.x;

			if (id < DYNAMIC_RASTERIZERS)
			{
				counts[i] = f(id);
				ids[i] = id;
			}
			else
			{
				counts[i] = 0;
				ids[i] = -1;
			}
		}
	}
};

template<int PER_THREAD_CHECK, int PER_THREAD_SORT, unsigned int DYNAMIC_RASTERIZERS, unsigned int NUM_THREADS>
struct PipelineSortingElements<PER_THREAD_CHECK, PER_THREAD_SORT, DYNAMIC_RASTERIZERS, NUM_THREADS, true>
{
	static constexpr int SORTING_ELEMENTS = PER_THREAD_SORT;

	template<typename F>
	__device__
	static void loadin(unsigned int(&counts)[SORTING_ELEMENTS], int(&ids)[SORTING_ELEMENTS], F f)
	{
		
		#pragma unroll
		for (int i = 0; i < SORTING_ELEMENTS; i++)
			counts[i] = 0;

		#pragma unroll
		for (int i = 0; i < PER_THREAD_CHECK; i++)
		{
			int id = threadIdx.x + i*NUM_THREADS;
			unsigned int count = 0;
			if (id < DYNAMIC_RASTERIZERS)
				count = f(id);
			#pragma unroll
			for (int j = 0; j < SORTING_ELEMENTS; j++)
			{
				if (count > counts[j])
				{
					unsigned int tc = count;
					int ti = id;
					count = counts[j];
					counts[j] = tc;
					id = ids[j];
					ids[j] = ti;
				}
			}
		}
	}
};



class VirtualRasterizerId
{
	__device__
	static int& r()
	{
		__shared__ int currentRasterizer;
		return currentRasterizer;
	}

public:
	__device__
	static int rasterizer()
	{
		return r();
	}

	__device__
	static void switchRasterizer(int i)
	{
		r() = i;
	}
};

template <bool ENABLE, unsigned int NUM_BLOCKS, unsigned int NUM_WARPS, unsigned int DYNAMIC_RASTERIZERS, class InputVertexAttributes, class PrimitiveType, class VertexShader, class CoverageShader, class FragmentShader, class BlendOp>
class Pipeline
{
	static constexpr int NUM_THREADS = NUM_WARPS * WARP_SIZE;
	static constexpr int VIRTUAL_RASTERIZER_TO_BLOCK_RATIO = (DYNAMIC_RASTERIZERS + NUM_THREADS - 1) / NUM_THREADS;
	
	static constexpr int MAX_LOCAL_SORT = 2;
	typedef PipelineSortingElements<VIRTUAL_RASTERIZER_TO_BLOCK_RATIO, MAX_LOCAL_SORT, DYNAMIC_RASTERIZERS, NUM_THREADS, false> Sorter; // (VIRTUAL_RASTERIZER_TO_BLOCK_RATIO > MAX_LOCAL_SORT) > Sorter;
	static constexpr int SORTING_ELEMENTS = Sorter::SORTING_ELEMENTS;
	static constexpr int SORT_MAX_BITS = 33 - static_clz<RASTERIZER_QUEUE_SIZE>::value;


	typedef PatternTileSpace<PATTERN_TECHNIQUE, DYNAMIC_RASTERIZERS, 8192, 8192, 8, 8, 8, 8, VirtualRasterizerId> UsedBinTileSpace;
	typedef BinTileRasterizationStage<DYNAMIC_RASTERIZERS, NUM_WARPS, BINRASTER_EXCLUSIVE_TILE_ACCESS_MODE, ENFORCE_PRIMITIVE_ORDER, FORCE_QUAD_SHADING, UsedBinTileSpace, CoverageShader, FragmentShader, FrameBuffer, BlendOp> RasterizationStage;
	typedef PerWarpPatchCachedGeometryStage<NUM_WARPS, InputVertexAttributes, PrimitiveType, VertexShader, FragmentShader, RasterizationStage, CLIPPING> GeometryStage;

	typedef cub::BlockRadixSort<unsigned int, NUM_THREADS, SORTING_ELEMENTS, int> VirtualRasterSorting;

	struct SharedState
	{
		union
		{
			__align__(16) char rasterization_stage_shared_memory[RasterizationStage::SHARED_MEMORY+1000];
			__align__(16) char geometry_stage_shared_memory[GeometryStage::SHARED_MEMORY+1000];
			__align__(16) typename VirtualRasterSorting::TempStorage virtualraster_sorting_storage;
		};
	};

	__device__
	static bool acquireRasterizer(typename VirtualRasterSorting::TempStorage& virtualraster_sorting_storage, bool only_efficient)
	{
		unsigned int counts[SORTING_ELEMENTS];
		int ids[SORTING_ELEMENTS];

		Sorter::loadin(counts, ids, [&](int id)
		{
			if (!virtual_rasterizers.isRasterizerActive(id))
			{
				int res = max(0, RasterizationStage::fillLevelNoCheck(id));
				return res;
			}
			return 0;
		});


		//TODO: set max sorting bits to queue size!
		//sort according to fill level
		VirtualRasterSorting(virtualraster_sorting_storage).SortDescending(counts, ids, 0, SORT_MAX_BITS);

		//try to acquire (TODO: add some kind of round robin?)
		for (int n = 0; n < NUM_THREADS; ++n)
		{
			bool found = false;

			if (threadIdx.x == n)
			{
				#pragma unroll
				for (int i = 0; i < SORTING_ELEMENTS; ++i)
				{
					unsigned int threshold = only_efficient ? (DYNAMIC_RASTERIZER_EFFICIENT_THRESHOLD ? DYNAMIC_RASTERIZER_EFFICIENT_THRESHOLD : NUM_THREADS) : 1;
					if (counts[i] >= threshold && ids[i] >= 0)
					{
						if (virtual_rasterizers.setRasterizerActive(ids[i]))
						{
							VirtualRasterizerId::switchRasterizer(ids[i]);
							found = true;
							break;
						}
					}
				}
			}

			if (__syncthreads_or(found))
				return true;
		}
		return false;
	}

	__device__
	static void freeMyRasterizer()
	{
		if (threadIdx.x == 0)
		{
			virtual_rasterizers.setRasterizerInactive(VirtualRasterizerId::rasterizer());
		}
	}

public:
	__device__
	static void run()
	{
		__shared__ SharedState shared_memory;

		Instrumentation::BlockObserver<0, 0> observer;

		__shared__ volatile int runstate[5];
		runstate[0] = true;
		runstate[3] = false;
		runstate[4] = false;
		__syncthreads();

		//runstate:
		// 0 geometry active
		// 1 rasterizer active
		// 2 rasterizer efficient
		// 3 can not run geometry
		// 4 rasterizer aquired

		while (runstate[0] || runstate[1])
		{
			if (runstate[0] && !runstate[4])
			{
				RasterizationStage::writeIterateCanNotReceiveAllNoSync(&runstate[3]);
				__syncthreads();
				if (!runstate[3])
				{
					if (!GeometryStage::run(shared_memory.geometry_stage_shared_memory))
					{
						if (threadIdx.x == 0)
						{
							atomicSub(&geometryProducingBlocksCount, 1);
							runstate[0] = false;
						}
					}
				}
			}
			__syncthreads();

			if (!runstate[4])
			{
				runstate[1] = ldg_cg(&geometryProducingBlocksCount) != 0;
				__syncthreads();
				runstate[4] = acquireRasterizer(shared_memory.virtualraster_sorting_storage, runstate[0] && !runstate[3]);
			}
			if (runstate[4])
			{
				runstate[1] = true;
				RasterizationStage::prepareRun(shared_memory.rasterization_stage_shared_memory, &runstate[2]);
				__syncthreads();
				if (!RasterizationStage::run(shared_memory.rasterization_stage_shared_memory))
				{
					runstate[4] = false;
					freeMyRasterizer();
				}
			}
			
			runstate[3] = false;
			__syncthreads();
		}
	}
};
#endif  // INCLUDED_CURE_PIPELINE
