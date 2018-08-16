


#ifndef INCLUDED_CURE_PIPELINE_MODULE
#define INCLUDED_CURE_PIPELINE_MODULE

#pragma once

#include "config.h"
#include "Pipeline.cuh"


namespace cuRE
{
	template <template <unsigned int, unsigned int, unsigned int, unsigned int> class PipelineEntryPoint, class PipelineConfigs>
	class PipelineInstantiator;

	template <template <unsigned int, unsigned int, unsigned int, unsigned int> class PipelineEntryPoint>
	class PipelineInstantiator<PipelineEntryPoint, PipelineConfigList<>>
	{
		template <template <unsigned int, unsigned int, unsigned int, unsigned int> class PipelineEntryPoint, class PipelineConfigs>
		friend class PipelineInstantiator;

		__host__
		static void cause_instantiation()
		{
		}
	};

	template <template <unsigned int, unsigned int, unsigned int, unsigned int> class PipelineEntryPoint, unsigned int NUM_MULTIPROCESSORS, unsigned int BLOCKS_PER_MULTIPROCESSOR, unsigned int WARPS_PER_BLOCK, unsigned int DYNAMIC_RASTERIZERS, unsigned int... REM_NUM_MULTIPROCESSORS, unsigned int... REM_BLOCKS_PER_MULTIPROCESSOR, unsigned int... REM_WARPS_PER_BLOCK, unsigned int... REM_VIRTUAL_RASTERIZERS>
	class PipelineInstantiator<PipelineEntryPoint, PipelineConfigList<PipelineConfig<NUM_MULTIPROCESSORS, BLOCKS_PER_MULTIPROCESSOR, WARPS_PER_BLOCK, DYNAMIC_RASTERIZERS>, PipelineConfig<REM_NUM_MULTIPROCESSORS, REM_BLOCKS_PER_MULTIPROCESSOR, REM_WARPS_PER_BLOCK, REM_VIRTUAL_RASTERIZERS>...>>
	{
		template <template <unsigned int, unsigned int, unsigned int, unsigned int> class PipelineEntryPoint, class PipelineConfigs>
		friend class PipelineInstantiator;

		__host__
		static void cause_instantiation()
		{
			PipelineEntryPoint<NUM_MULTIPROCESSORS, BLOCKS_PER_MULTIPROCESSOR, WARPS_PER_BLOCK, DYNAMIC_RASTERIZERS>::cause_instantiation();
			PipelineInstantiator<PipelineEntryPoint, PipelineConfigList<PipelineConfig<REM_NUM_MULTIPROCESSORS, REM_BLOCKS_PER_MULTIPROCESSOR, REM_WARPS_PER_BLOCK, REM_VIRTUAL_RASTERIZERS>...>>::cause_instantiation();
		}
	};
}

#define PIPELINE_ENTRY_POINT(NAME, ENABLE, VertexBuffer, PrimitiveType, VertexShader, CoverageShader, FragmentShader, BlendOp) \
	namespace Pipelines \
	{ \
		template <unsigned int NUM_MULTIPROCESSORS, unsigned int BLOCKS_PER_MULTIPROCESSOR, unsigned int WARPS_PER_BLOCK, unsigned int DYNAMIC_RASTERIZERS> \
		__launch_bounds__(WARPS_PER_BLOCK * WARP_SIZE, BLOCKS_PER_MULTIPROCESSOR) \
		__global__ \
		void NAME() \
		{ \
			assert(gridDim.x == NUM_MULTIPROCESSORS * BLOCKS_PER_MULTIPROCESSOR && gridDim.y == 1 && gridDim.z == 1 && \
			       blockDim.x == WARPS_PER_BLOCK * WARP_SIZE && blockDim.y == 1 && blockDim.z == 1 && \
			       warpSize == WARP_SIZE); \
			Pipeline<ENABLE, NUM_MULTIPROCESSORS * BLOCKS_PER_MULTIPROCESSOR, WARPS_PER_BLOCK, DYNAMIC_RASTERIZERS, VertexBuffer, PrimitiveType, VertexShader, CoverageShader, FragmentShader, BlendOp>::run(); \
		} \
		template <unsigned int NUM_MULTIPROCESSORS, unsigned int BLOCKS_PER_MULTIPROCESSOR, unsigned int WARPS_PER_BLOCK, unsigned int DYNAMIC_RASTERIZERS> \
		class NAME##EntryPoint \
		{ \
			template <template <unsigned int, unsigned int, unsigned int, unsigned int> class PipelineEntryPoint, class PipelineConfigs> \
			friend class cuRE::PipelineInstantiator; \
			__host__ static void cause_instantiation() { NAME<NUM_MULTIPROCESSORS, BLOCKS_PER_MULTIPROCESSOR, WARPS_PER_BLOCK, DYNAMIC_RASTERIZERS><<<1,1>>>(); } \
		}; \
	}

#define INSTANTIATE_PIPELINE(NAME, ENABLE, VertexBuffer, PrimitiveType, VertexShader, CoverageShader, FragmentShader, BlendOp) \
	PIPELINE_ENTRY_POINT(NAME, ENABLE, VertexBuffer, PrimitiveType, VertexShader, CoverageShader, FragmentShader, BlendOp); \
	template class cuRE::PipelineInstantiator<Pipelines::NAME##EntryPoint, PipelineConfigs>

#endif  // INCLUDED_CURE_PIPELINE_MODULE
